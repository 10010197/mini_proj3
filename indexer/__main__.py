"""② 인덱싱 계층 실행: docs.db(in_kb=1) → 파싱·정제 → 청킹(설정별) → 청크 중복 처리 → (선택) 임베딩.

    python -m indexer chunk                       # 모든 청킹 설정의 청크 파일 + 길이 분포 통계
    python -m indexer chunk --configs structure
    python -m indexer embed --configs structure   # 청크 임베딩 (캐시 재사용, OpenAI 비용 발생)

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m indexer")
    ap.add_argument("command", choices=["chunk", "embed"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "chunk":
        print(json.dumps(build_chunks(args.configs), ensure_ascii=False, indent=2))
    else:
        from indexer.embed import DEFAULT_MODEL, embed_texts

        model = args.model or DEFAULT_MODEL
        for cfg in args.configs:
            chunks = read_chunks(cfg)
            vecs = embed_texts([c["text"] for c in chunks], model)
            log.info("%s: %d개 임베딩 완료 %s", cfg, len(vecs), np.asarray(vecs).shape)


if __name__ == "__main__":
    main()
