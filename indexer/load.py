"""② 인덱싱 계층 — 벡터 DB(Chroma) 적재 및 적재 검증.

    청크 파일(store/chunks/{config}.jsonl) + 임베딩 캐시 → store/chroma/ (PersistentClient, 디스크 영속화)

- 컬렉션 이름: `{config}__{model}` (예: structure__text-embedding-3-small). 거리: cosine.
- 벡터 DB에는 **청크와 메타데이터만** 넣는다. 원문 전체는 store/docs.db에 두고, 챗봇이 원문을 보여줄 때
  메타데이터의 doc_id로 따로 조회한다.
- document = 청크 `text`(머리말 + 본문, 임베딩한 그 텍스트). 답변 인용용 본문은 머리말 다음 줄부터다.
- 벡터는 임베딩 캐시에서만 읽는다(cache_only). 적재 단계에서는 API를 부르지 않는다.
- 다시 적재하면 컬렉션을 지우고 새로 만든다(전체 재구축). 챗봇은 이 컬렉션을 읽기만 한다.

적재 검증 (store/vector_load_stats.json)
  1. 개수   컬렉션 count = 청크 파일 줄 수, ID 집합 일치, 고유 doc_id 수·출처별 청크 수 일치
  2. 빈 청크 document가 비었거나 메타데이터 필수 항목(doc_id·title·url·published_at·source_type)이 빈 청크 0
  3. 벡터   무작위 청크 200개의 저장 벡터 = 캐시 벡터 (차원·값)
  4. 영속화 클라이언트를 새로 열어 다시 읽었을 때 count가 같음
  5. 샘플 검색 평가셋 질문 몇 개로 상위 5개 청크를 뽑아 정답 문서가 있는지와 결과를 기록 (질의 임베딩만 API 호출)
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np

from collector import storage
from indexer.chunk import METADATA_FIELDS

log = logging.getLogger("indexer.load")

CHROMA_DIR = storage.PROJECT_ROOT / "store" / "chroma"
LOAD_STATS = storage.PROJECT_ROOT / "store" / "vector_load_stats.json"
EVAL_SET = storage.PROJECT_ROOT / "eval" / "eval_set.jsonl"
BATCH = 5000                      # Chroma 1회 add 상한(약 5,461)보다 작게
REQUIRED_META = ("doc_id", "title", "url", "published_at", "source_type")
SAMPLE_QUERY_IDS = ("F01", "F13", "T01", "C01", "S01", "U01")   # 사실·시점·비교·종합·답변 불가


def collection_name(config: str, model: str) -> str:
    return f"{config}__{model}"


def client(path: Path = CHROMA_DIR):
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _meta(c: dict) -> dict:
    out = {}
    for k in METADATA_FIELDS:
        v = c.get(k)
        out[k] = "" if v is None else (int(v) if isinstance(v, (bool, np.integer)) else v)
    return out


def load_config(config: str, model: str) -> dict:
    from indexer.__main__ import read_chunks
    from indexer.embed import embed_texts

    chunks = read_chunks(config)
    vecs = embed_texts([c["text"] for c in chunks], model, cache_only=True)
    cl = client()
    name = collection_name(config, model)
    if name in [c.name if hasattr(c, "name") else c for c in cl.list_collections()]:
        cl.delete_collection(name)
    col = cl.create_collection(name, metadata={
        "hnsw:space": "cosine", "embedding_model": model, "chunk_config": config,
        "dim": int(vecs.shape[1]), "built_at": storage.now_iso(),
    })
    for i in range(0, len(chunks), BATCH):
        part = chunks[i:i + BATCH]
        col.add(ids=[c["chunk_id"] for c in part],
                embeddings=vecs[i:i + BATCH].tolist(),
                documents=[c["text"] for c in part],
                metadatas=[_meta(c) for c in part])
        log.info("  %s: %d/%d 적재", name, min(i + BATCH, len(chunks)), len(chunks))
    return verify_load(config, model, chunks, vecs)


def verify_load(config: str, model: str, chunks: list[dict] | None = None, vecs: np.ndarray | None = None) -> dict:
    from indexer.__main__ import read_chunks
    from indexer.embed import embed_texts

    chunks = chunks if chunks is not None else read_chunks(config)
    name = collection_name(config, model)
    col = client().get_collection(name)

    # 1·2. 개수와 빈 청크 — 컬렉션 전체를 페이지로 읽는다
    ids, empty_docs, missing_meta, src = set(), 0, Counter(), Counter()
    doc_ids = set()
    page = 5000
    for off in range(0, col.count(), page):
        got = col.get(limit=page, offset=off, include=["documents", "metadatas"])
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]):
            ids.add(i)
            if not (d or "").strip():
                empty_docs += 1
            for k in REQUIRED_META:
                if m.get(k) in (None, ""):
                    missing_meta[k] += 1
            src[m.get("source_type", "")] += 1
            doc_ids.add(m.get("doc_id"))
    file_ids = {c["chunk_id"] for c in chunks}
    file_src = Counter(c["source_type"] for c in chunks)

    # 3. 저장 벡터 = 캐시 벡터
    rnd = random.Random(42)
    sample = rnd.sample(chunks, min(200, len(chunks)))
    stored = col.get(ids=[c["chunk_id"] for c in sample], include=["embeddings"])
    by_id = dict(zip(stored["ids"], stored["embeddings"]))
    ref = embed_texts([c["text"] for c in sample], model, cache_only=True)
    vec_mismatch = sum(1 for c, r in zip(sample, ref)
                       if c["chunk_id"] not in by_id or not np.allclose(np.asarray(by_id[c["chunk_id"]]), r, atol=1e-5))
    dim = len(next(iter(by_id.values()))) if by_id else 0

    # 4. 영속화 — 새 클라이언트로 다시 열기
    reopened = client().get_collection(name).count()

    # 5. 샘플 검색
    items = {json.loads(x)["id"]: json.loads(x) for x in EVAL_SET.read_text(encoding="utf-8").splitlines() if x.strip()}
    qs = [items[q] for q in SAMPLE_QUERY_IDS if q in items]
    qv = embed_texts([q["q"] for q in qs], model)
    samples = []
    for q, v in zip(qs, qv):
        res = col.query(query_embeddings=[v.tolist()], n_results=5, include=["metadatas", "distances"])
        top = [{"rank": r + 1, "sim": round(1 - d, 3), "doc_id": m["doc_id"], "source_type": m["source_type"],
                "published_at": m["published_at"][:10], "title": m["title"][:70], "section": m["section"][:40]}
               for r, (m, d) in enumerate(zip(res["metadatas"][0], res["distances"][0]))]
        gold = set(q.get("gold_doc_ids") or [])
        samples.append({"id": q["id"], "type": q["type"], "q": q["q"], "gold": sorted(gold),
                        "gold_in_top5": bool(gold & {t["doc_id"] for t in top}) if gold else None, "top5": top})

    out = {
        "collection": name, "path": str(CHROMA_DIR.relative_to(storage.PROJECT_ROOT)), "checked_at": storage.now_iso(),
        "chunks_in_file": len(chunks), "chunks_in_db": len(ids), "count_after_reopen": reopened,
        "ids_missing_in_db": len(file_ids - ids), "ids_extra_in_db": len(ids - file_ids),
        "docs_in_db": len(doc_ids), "docs_in_file": len({c["doc_id"] for c in chunks}),
        "by_source_db": dict(sorted(src.items())), "by_source_file": dict(sorted(file_src.items())),
        "empty_documents": empty_docs, "missing_required_meta": dict(missing_meta),
        "vector_dim": dim, "vector_sample": len(sample), "vector_mismatch": vec_mismatch,
        "sample_search": samples,
    }
    out["ok"] = (len(ids) == len(chunks) == reopened and not out["ids_missing_in_db"] and not out["ids_extra_in_db"]
                 and out["docs_in_db"] == out["docs_in_file"] and src == file_src
                 and empty_docs == 0 and not missing_meta and vec_mismatch == 0)
    stats = json.loads(LOAD_STATS.read_text(encoding="utf-8")) if LOAD_STATS.exists() else {}
    stats[name] = out
    LOAD_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("%s 적재 검증: %s | 청크 파일 %d · DB %d · 재오픈 %d | 문서 %d | 빈 청크 %d | 필수 메타 누락 %s | 벡터 불일치 %d/%d (dim %d)",
             name, "OK" if out["ok"] else "FAIL", len(chunks), len(ids), reopened, len(doc_ids),
             empty_docs, dict(missing_meta) or 0, vec_mismatch, len(sample), dim)
    for s in samples:
        log.info("  [%s] %s  (정답 top5: %s)", s["id"], s["q"][:50], s["gold_in_top5"])
        for t in s["top5"][:3]:
            log.info("      %d. %.3f %-10s %s %s", t["rank"], t["sim"], t["source_type"], t["published_at"], t["title"])
    return out
