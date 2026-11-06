"""④ 챗봇 계층 — 검색: Chroma(store/chroma, 읽기 전용) 조회 + 문서 단위 정리 + 원문 조회.

- 인덱서가 만든 컬렉션을 **읽기만** 한다. 문서 임베딩·인덱스 쓰기·크롤링은 하지 않는다.
  질의 임베딩만 인덱서와 같은 모델(text-embedding-3-small)로 만든다.
- 청크를 넉넉히(POOL) 가져와 문서당 최대 MAX_CHUNKS_PER_DOC개만 남기고 상위 K_DOCS개 문서로 자른다.
  (같은 문서의 청크가 상위를 독차지하는 문제: docs/indexing.md §5.2)
- 원문 전체는 벡터 DB에 없다. 사용자가 원문을 요청하면 get_document(doc_id)로 store/docs.db를 조회한다.
- 검색 방식(RETRIEVAL_MODE): dense(벡터) · hybrid(벡터 + BM25, RRF 결합) · hybrid_time(+질문 연도로 거른 벡터 검색).
  BM25 인덱스(store/bm25/)도 인덱서가 만들고 여기서는 읽기만 한다(rag/bm25.py). 비교 결과: docs/evaluation.md §6.
  hybrid_rerank: hybrid_time 후보를 넓게 모아 LLM이 재정렬한다(rag/rerank.py). 비교 결과: docs/evaluation.md §7.

설정(.env, 선택): EMBED_MODEL, CHUNK_CONFIG, RETRIEVAL_MODE — 기본값은 인덱서와 같다.
"""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "store" / "chroma"
DOCS_DB = PROJECT_ROOT / "store" / "docs.db"
ENV = {**dotenv_values(PROJECT_ROOT / ".env"), **{k: v for k, v in os.environ.items() if k in ("EMBED_MODEL", "CHUNK_CONFIG")}}

EMBED_MODEL = ENV.get("EMBED_MODEL") or "text-embedding-3-small"
CHUNK_CONFIG = ENV.get("CHUNK_CONFIG") or "structure"
COLLECTION = f"{CHUNK_CONFIG}__{EMBED_MODEL}"


MODES = ("dense", "hybrid", "hybrid_time", "hybrid_rerank")   # hybrid_rerank = hybrid_time 후보 + LLM 재정렬 (rag/rerank.py)
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE") or ENV.get("RETRIEVAL_MODE") or "hybrid_time"   # 근거: docs/evaluation.md §6
RRF_K = 60                # Reciprocal Rank Fusion 상수


def configure(chunk_config: str | None = None, embed_model: str | None = None, mode: str | None = None) -> str:
    """사용할 컬렉션·검색 방식을 바꾼다 (평가 하네스의 설정 비교용). 챗봇은 기본값을 쓴다."""
    global CHUNK_CONFIG, EMBED_MODEL, COLLECTION, RETRIEVAL_MODE
    CHUNK_CONFIG = chunk_config or CHUNK_CONFIG
    EMBED_MODEL = embed_model or EMBED_MODEL
    RETRIEVAL_MODE = mode or RETRIEVAL_MODE
    COLLECTION = f"{CHUNK_CONFIG}__{EMBED_MODEL}"
    vectorstore.cache_clear()
    return COLLECTION


POOL = 40                 # 1차로 가져오는 청크 수 (목록마다)
K_DOCS = 6                # 답변 근거로 넘기는 문서 수
MAX_CHUNKS_PER_DOC = 2    # 문서당 청크 상한
RERANK_DOCS = 15          # hybrid_rerank: 재정렬 대상으로 모으는 문서 수 (여기서 K_DOCS개를 고른다)
RERANK_POOL = 80          # hybrid_rerank: 1차 청크 후보 수 (목록마다)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float          # 코사인 유사도 (1 - 거리)
    text: str             # 청크 text (머리말 + 본문)
    meta: dict


@lru_cache(maxsize=1)
def vectorstore():
    import chromadb
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    if not CHROMA_DIR.exists():
        raise RuntimeError("store/chroma/가 없습니다. 먼저 `python -m indexer load`로 인덱스를 적재하세요.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    names = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if COLLECTION not in names:
        raise RuntimeError(f"컬렉션 {COLLECTION}이 없습니다 (있는 것: {names}).")
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=ENV.get("OPENAI_API_KEY"))
    return Chroma(client=client, collection_name=COLLECTION, embedding_function=emb)


def build_where(source_types: list[str] | None = None, year_from: int | None = None,
                year_to: int | None = None, topic: str | None = None) -> dict | None:
    """UI 필터 → Chroma 메타데이터 조건. 조건이 없으면 None.
    topic은 'ADC; 신약 기술수출'처럼 여러 값이 든 문자열이라 Chroma where로 못 거르므로 search()에서 후처리한다."""
    conds: list[dict] = []
    if source_types:
        conds.append({"source_type": {"$in": list(source_types)}})
    if year_from:
        conds.append({"year": {"$gte": int(year_from)}})
    if year_to:
        conds.append({"year": {"$lte": int(year_to)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def match_where(meta: dict, where: dict | None) -> bool:
    """build_where가 만드는 조건($and·$in·$gte·$lte·$eq)을 메타데이터 dict에 적용 (BM25 후보 거르기용)."""
    if not where:
        return True
    if "$and" in where:
        return all(match_where(meta, w) for w in where["$and"])
    for field, cond in where.items():
        v = meta.get(field)
        ops = cond if isinstance(cond, dict) else {"$eq": cond}
        for op, x in ops.items():
            if op == "$in" and v not in x or op == "$eq" and v != x \
                    or op == "$gte" and not (v is not None and v >= x) or op == "$lte" and not (v is not None and v <= x):
                return False
    return True


YEAR_IN_QUERY = re.compile(r"(20\d{2})\s*년")
KB_YEARS = range(2021, 2027)


def query_years(query: str) -> list[int]:
    """질문에 적힌 연도(지식베이스 기간 안) — '2024년 10월', '2025년 매출 전망' 등."""
    return sorted({int(y) for y in YEAR_IN_QUERY.findall(query) if int(y) in KB_YEARS})


def _and(*conds: dict | None) -> dict | None:
    cs = [c for c in conds if c]
    return None if not cs else cs[0] if len(cs) == 1 else {"$and": cs}


def _dense(qv: list[float], k: int, where: dict | None) -> list[tuple[str, dict, str, float]]:
    res = vectorstore()._collection.query(query_embeddings=[qv], n_results=k, where=where,
                                          include=["documents", "metadatas", "distances"])
    return [(i, m, d, 1 - float(dist)) for i, m, d, dist in
            zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0])]


def _keyword(query: str, qv: list[float], k: int, where: dict | None) -> list[tuple[str, dict, str, float]]:
    from rag import bm25

    ranked = bm25.load(CHUNK_CONFIG).search(query, k * 3 if where else k)
    if not ranked:
        return []
    got = vectorstore()._collection.get(ids=[c for c, _ in ranked], include=["documents", "metadatas", "embeddings"])
    by_id = {i: (m, d, e) for i, m, d, e in zip(got["ids"], got["metadatas"], got["documents"], got["embeddings"])}
    q = np.asarray(qv, dtype=np.float32)
    out = []
    for cid, _ in ranked:
        if cid in by_id and match_where(by_id[cid][0], where):
            m, d, e = by_id[cid]
            out.append((cid, m, d, float(np.dot(q, np.asarray(e, dtype=np.float32)))))   # 표시용 코사인 (벡터는 정규화돼 있음)
        if len(out) >= k:
            break
    return out


def candidates(query: str, k: int, where: dict | None, mode: str) -> list[tuple[str, dict, str, float]]:
    """mode별 청크 후보 (순위순). 여러 목록은 RRF(1/(RRF_K+순위)의 합)로 합친다. 반환 점수는 코사인 유사도(표시용)."""
    qv = vectorstore().embeddings.embed_query(query)
    lists = [_dense(qv, k, where)]
    if mode in ("hybrid", "hybrid_time"):
        lists.append(_keyword(query, qv, k, where))
    if mode == "hybrid_time" and (years := query_years(query)):
        lists.append(_dense(qv, k, _and(where, {"year": {"$in": years}})))
    if len(lists) == 1:
        return lists[0]
    fused: dict[str, float] = {}
    info: dict[str, tuple] = {}
    for lst in lists:
        for r, item in enumerate(lst, 1):
            fused[item[0]] = fused.get(item[0], 0.0) + 1.0 / (RRF_K + r)
            info.setdefault(item[0], item)
    return [info[c] for c in sorted(fused, key=fused.get, reverse=True)]


def search(query: str, k_docs: int = K_DOCS, max_chunks_per_doc: int = MAX_CHUNKS_PER_DOC,
           pool: int = POOL, where: dict | None = None, topic: str | None = None, mode: str | None = None) -> list[Hit]:
    """질의 → 청크 후보(mode) → 문서 단위로 묶어 상위 k_docs개 문서(문서당 최대 max_chunks_per_doc 청크).
    반환 순서: 문서 순위(그 문서의 최고 순위 청크 기준) → 문서 안에서는 순위순.
    mode: dense(벡터만) · hybrid(벡터+BM25, RRF) · hybrid_time(+질문 연도로 거른 벡터 검색). 기본은 RETRIEVAL_MODE.
    where: Chroma 메타데이터 조건(build_where), topic: 하위 주제 태그 후처리 필터(이때는 pool을 3배로 넓힌다).
    청크가 많은 긴 문서(예: 정책 붙임의 의약품 목록)가 pool을 채워 문서가 k_docs개보다 적게 모이면 pool을 5배로 한 번 더 조회한다.
    hybrid_rerank: hybrid_time 후보에서 문서 RERANK_DOCS개를 모아 LLM으로 재정렬한 뒤 k_docs개를 고른다."""
    mode = mode or RETRIEVAL_MODE
    rerank = mode == "hybrid_rerank"
    base = "hybrid_time" if rerank else mode
    n_docs = max(RERANK_DOCS, k_docs) if rerank else k_docs
    k = max(pool, RERANK_POOL) if rerank else pool
    k = k * 3 if topic else k
    hits = _group(candidates(query, k, where, base), n_docs, max_chunks_per_doc, topic)
    if len({h.doc_id for h in hits}) < k_docs:
        hits = _group(candidates(query, k * 5, where, base), n_docs, max_chunks_per_doc, topic)
    if rerank:
        from rag import rerank as rr

        hits = rr.rerank(query, hits, k_docs)
    return hits


def _group(res: list, k_docs: int, max_chunks_per_doc: int, topic: str | None) -> list[Hit]:
    per_doc: dict[str, list[Hit]] = {}
    order: list[str] = []
    for cid, m, text, score in res:
        if topic and topic not in str(m.get("topics", "")).split("; "):
            continue
        did = m.get("doc_id", "")
        if did not in per_doc:
            if len(order) >= k_docs:
                continue
            per_doc[did] = []
            order.append(did)
        if len(per_doc[did]) < max_chunks_per_doc:
            per_doc[did].append(Hit(m.get("chunk_id", cid), did, round(score, 4), text, dict(m)))
    return [h for d in order for h in per_doc[d]]


def get_document(doc_id: str) -> dict | None:
    """원문 조회 (store/docs.db, 읽기 전용)."""
    con = sqlite3.connect(f"file:{DOCS_DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT doc_id, source_type, evidence_role, title, body, url, published_at, publisher, loaded_at "
                          "FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()
