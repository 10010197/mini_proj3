"""② 인덱싱 계층 실행: docs.db(in_kb=1) → 파싱·정제 → 청킹(설정별) → 청크 중복 처리 → (선택) 임베딩.

    python -m indexer chunk                       # 모든 청킹 설정의 청크 파일 + 길이 분포 통계
    python -m indexer chunk --configs structure
    python -m indexer embed --configs structure   # 청크 임베딩 (캐시 재사용, OpenAI 비용 발생)
    python -m indexer verify                      # 문서 수·청크 수·임베딩 수 대조 (API 호출 없음, 실패 시 종료 코드 1)
    python -m indexer load                        # structure 청크 → store/chroma/ 적재 + 적재 검증 (indexer/load.py)
    python -m indexer verify-load                 # 적재 검증만 다시 (개수·빈 청크·벡터·영속화·샘플 검색)

chunk·embed는 끝날 때 verify를 자동으로 실행한다. 검증 결과는 chunk_stats.json의 {config}.verify에 남는다.

산출물 (store/, git 제외)
    store/chunks/{config}.jsonl        청크 텍스트 + 메타데이터
    store/chunks/chunk_stats.json      설정별 청크 수·길이 분포·중복 제거 건수
    store/embeddings/{model}/          임베딩 캐시 (text_hash → 벡터)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from collector import storage
from indexer.chunk import CONFIGS, chunk_doc
from indexer.dedup import dedup_chunks

log = logging.getLogger("indexer")
CHUNK_DIR = storage.PROJECT_ROOT / "store" / "chunks"


def load_kb(db_path: Path = storage.DEFAULT_DB) -> list[dict]:
    with storage.connect(db_path) as con:
        has_dedup = con.execute("SELECT name FROM sqlite_master WHERE name='news_dedup'").fetchone()
        join = "LEFT JOIN news_dedup n USING(doc_id)" if has_dedup else ""
        ev = "n.event_id" if has_dedup else "NULL"
        df = pd.read_sql(
            f"SELECT d.doc_id, d.source_type, d.evidence_role, d.origin, d.title, d.body, d.url, d.published_at, "
            f"d.publisher, d.topics, d.is_outlook, d.meta, {ev} AS event_id FROM documents d {join} WHERE d.in_kb = 1",
            con,
        )
    return df.to_dict("records")


def length_stats(chunks: list[dict]) -> dict:
    s = pd.Series([c["char_len"] for c in chunks])
    by = pd.DataFrame(chunks).groupby("source_type")["char_len"]
    return {
        "chunks": len(chunks),
        "docs": len({c["doc_id"] for c in chunks}),
        "chars_total": int(s.sum()),
        "len_pct": {k: int(v) for k, v in s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()},
        "under_150": int((s < 150).sum()),
        "over_1500": int((s > 1500).sum()),
        "by_source": {k: {"chunks": int(v.size), "median": int(v.median()), "p95": int(v.quantile(0.95))} for k, v in by},
        "chunks_per_doc_median": float(pd.Series([c["n_chunks"] for c in chunks if c["chunk_index"] == 0]).median()),
    }


def build_chunks(configs: list[str], db_path: Path = storage.DEFAULT_DB) -> dict:
    docs = load_kb(db_path)
    log.info("지식베이스 문서 %d건", len(docs))
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = CHUNK_DIR / "chunk_stats.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for cfg in configs:
        chunks = [c for d in docs for c in chunk_doc(d, cfg)]
        kept, dstats = dedup_chunks(chunks)
        with open(CHUNK_DIR / f"{cfg}.jsonl", "w", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        alias = dstats.pop("doc_alias")
        (CHUNK_DIR / f"{cfg}_doc_alias.json").write_text(json.dumps(alias, ensure_ascii=False), encoding="utf-8")
        all_stats[cfg] = {**length_stats(kept), "dedup": dstats, "generated_at": storage.now_iso()}
        log.info("%s: 청크 %d (중복 처리 %s)", cfg, len(kept), dstats["dropped"])
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_stats


def read_chunks(config: str) -> list[dict]:
    with open(CHUNK_DIR / f"{config}.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


# --------------------------------------------------------------------------- #
# 인덱싱 검증: 문서 수 · 청크 수 · 임베딩 수 대조
# --------------------------------------------------------------------------- #
LOW_COVERAGE = 0.3      # 원문 대비 청크 본문 비율이 이보다 낮으면 '파싱 손실 의심'으로 보고


def verify_counts(docs: list[dict], chunks: list[dict], alias: dict, model: str | None = None) -> dict:
    """지식베이스 문서 = 청크가 있는 문서 + 내용 중복으로만 빠진 문서(alias)인지, 청크 번호·ID가 온전한지,
    (model을 주면) 모든 청크가 임베딩 캐시에 있는지 확인한다. 결과의 ok가 False면 인덱스에 구멍이 있다."""
    kb = {d["doc_id"]: d for d in docs}
    per_doc: dict[str, list[dict]] = {}
    for c in chunks:
        per_doc.setdefault(c["doc_id"], []).append(c)
    missing = set(kb) - set(per_doc)
    by_alias = {d for d in missing if d in alias}
    unexplained = sorted(missing - by_alias)
    not_in_kb = sum(len(v) for d, v in per_doc.items() if d not in kb)
    ids = [c["chunk_id"] for c in chunks]
    index_gaps = sum(1 for v in per_doc.values()
                     if sorted(c["chunk_index"] for c in v) != list(range(len(v))) or any(c["n_chunks"] != len(v) for c in v))
    low = sorted(d for d, v in per_doc.items()
                 if d in kb and sum(len(c["body"]) for c in v) < LOW_COVERAGE * len(kb[d]["body"] or ""))

    def by_source(ids_: set | list) -> dict:
        return dict(pd.Series([kb[d]["source_type"] for d in ids_ if d in kb], dtype=object).value_counts().sort_index())

    out = {
        "kb_docs": len(kb),
        "chunked_docs": len(set(per_doc) & set(kb)),
        "docs_dropped_as_duplicate": len(by_alias),
        "docs_unexplained_missing": len(unexplained),
        "unexplained_sample": unexplained[:10],
        "chunks": len(chunks),
        "duplicate_chunk_ids": len(ids) - len(set(ids)),
        "empty_chunks": sum(1 for c in chunks if not (c.get("body") or "").strip() or not (c.get("text") or "").strip()),
        "chunks_not_in_kb": not_in_kb,
        "docs_with_index_gaps": index_gaps,
        "docs_low_coverage": len(low),
        "low_coverage_sample": low[:10],
        "by_source": {s: {"kb_docs": n, "chunked_docs": by_source(set(per_doc) & set(kb)).get(s, 0),
                          "chunks": sum(1 for c in chunks if c["source_type"] == s)}
                      for s, n in by_source(list(kb)).items()},
    }
    ok = not (unexplained or out["duplicate_chunk_ids"] or out["empty_chunks"] or not_in_kb or index_gaps)
    if model:
        from indexer.embed import count_cached

        texts = [c["text"] for c in chunks]
        n_uniq = len(set(texts))
        cached = count_cached(texts, model)
        out["embedding"] = {"model": model, "unique_texts": n_uniq, "cached": cached, "missing": n_uniq - cached}
        ok = ok and cached == n_uniq
    out["ok"] = bool(ok)
    return out


def run_verify(configs: list[str], model: str | None, db_path: Path = storage.DEFAULT_DB) -> bool:
    docs = load_kb(db_path)
    stats_path = CHUNK_DIR / "chunk_stats.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    all_ok = True
    for cfg in configs:
        alias = json.loads((CHUNK_DIR / f"{cfg}_doc_alias.json").read_text(encoding="utf-8"))
        v = verify_counts(docs, read_chunks(cfg), alias, model)
        v["checked_at"] = storage.now_iso()
        all_stats.setdefault(cfg, {})["verify"] = v
        all_ok &= v["ok"]
        emb = v.get("embedding", {})
        log.info("%s: %s | 문서 %d = 청크 있음 %d + 중복으로 제외 %d + 설명 안 됨 %d | 청크 %d (빈 청크 %d) | 임베딩 누락 %s | 저커버리지 문서 %d",
                 cfg, "OK" if v["ok"] else "FAIL", v["kb_docs"], v["chunked_docs"], v["docs_dropped_as_duplicate"],
                 v["docs_unexplained_missing"], v["chunks"], v["empty_chunks"], emb.get("missing", "-"), v["docs_low_coverage"])
        if v["unexplained_sample"]:
            log.warning("  청크 없는 문서 예: %s", v["unexplained_sample"])
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    return all_ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m indexer")
    ap.add_argument("command", choices=["chunk", "embed", "verify", "load", "verify-load"])
    ap.add_argument("--configs", nargs="+", default=None, choices=list(CONFIGS),
                    help="기본: chunk·embed·verify는 전체, load·verify-load는 structure(챗봇이 쓰는 설정)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from indexer.embed import DEFAULT_MODEL

    model = args.model or DEFAULT_MODEL
    if args.configs is None:
        args.configs = ["structure"] if args.command in ("load", "verify-load") else list(CONFIGS)
    if args.command in ("load", "verify-load"):
        from indexer.load import load_config, verify_load

        ok = True
        for cfg in args.configs:
            if args.command == "load":
                if not run_verify([cfg], model):
                    log.error("%s: 청크·임베딩 검증 실패 → 적재하지 않음", cfg)
                    sys.exit(1)
                ok &= load_config(cfg, model)["ok"]
            else:
                ok &= verify_load(cfg, model)["ok"]
        if not ok:
            log.error("적재 검증 실패: store/vector_load_stats.json 확인")
            sys.exit(1)
        return
    if args.command == "chunk":
        build_chunks(args.configs)
        ok = run_verify(args.configs, model=None)
    elif args.command == "embed":
        from indexer.embed import embed_texts

        for cfg in args.configs:
            chunks = read_chunks(cfg)
            vecs = embed_texts([c["text"] for c in chunks], model)
            if len(vecs) != len(chunks):
                raise RuntimeError(f"{cfg}: 청크 {len(chunks)}개인데 벡터 {len(vecs)}개")
            log.info("%s: %d개 임베딩 완료 %s", cfg, len(vecs), np.asarray(vecs).shape)
        ok = run_verify(args.configs, model)
    else:
        ok = run_verify(args.configs, model)
    if not ok:
        log.error("검증 실패: store/chunks/chunk_stats.json 의 verify 항목 확인")
        sys.exit(1)


if __name__ == "__main__":
    main()
