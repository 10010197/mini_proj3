"""검색 평가: 청킹 설정(·임베딩 모델)별 Hit@k, MRR.

평가셋 eval/eval_set.jsonl 중 정답 문서가 있는 문항(답변 불가 제외)에 대해
질의 임베딩 → 청크 코사인 유사도 상위 → 문서 단위로 중복 제거한 순위에서 정답 문서 위치를 본다.

  Hit@k  상위 k개 문서 안에 정답 문서가 하나라도 있는 문항 비율
  MRR    첫 정답 문서 순위의 역수 평균 (상위 RANK_DEPTH개 밖이면 0)

정답 확장 (모든 설정에 똑같이 적용, 엄격 기준 Hit@3도 함께 보고):
  - 공시 정정본: 같은 회사·같은 제목의 원본과 [정정]본은 서로 정답
  - also_accept: 같은 발표를 다룬 다른 기사 중 답의 수치가 본문에 있음을 직접 확인한 문서
  - 뉴스 정답은 같은 사건(event_id)으로 묶인 지식베이스 기사도 정답으로 인정한다.
청크 중복 처리로 정답 문서의 청크가 다른 문서 쪽에 남은 경우(doc_alias)도 그 문서를 정답으로 인정한다.
(재배포본·동일 사건 기사는 내용이 겹쳐 어느 쪽이 검색돼도 같은 근거이기 때문. 근거: docs/kb_composition.md §1)

답변 불가 문항은 상위 1위 유사도만 기록해, 이후 '근거 없음' 판정 임계값을 정할 때 쓴다.

실행:
    python -m eval.retrieval_eval                                  # 모든 청킹 설정
    python -m eval.retrieval_eval --configs structure --model text-embedding-3-large
결과: eval/results/retrieval_{model}.json, 콘솔 표
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from collector import storage
from indexer.__main__ import read_chunks
from indexer.chunk import CONFIGS
from indexer.embed import DEFAULT_MODEL, embed_texts

ROOT = storage.PROJECT_ROOT
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
OUT_DIR = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)
RANK_DEPTH = 20
CHUNK_POOL = 200          # 문서 단위 순위를 만들기 위해 가져오는 청크 수


def load_eval() -> list[dict]:
    return [json.loads(x) for x in EVAL_SET.read_text(encoding="utf-8").splitlines() if x.strip()]


def event_siblings(db_path: Path = storage.DEFAULT_DB) -> dict[str, set[str]]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT n.doc_id, n.event_id FROM news_dedup n JOIN documents d USING(doc_id) "
            "WHERE n.event_id IS NOT NULL AND d.in_kb = 1").fetchall()
    by_event: dict[str, set[str]] = defaultdict(set)
    for d, e in rows:
        by_event[e].add(d)
    return {d: by_event[e] for d, e in rows}


def correction_aliases(db_path: Path = storage.DEFAULT_DB) -> dict[str, set[str]]:
    """공시 정정본: 같은 회사·같은 제목(앞의 '[정정] ' 제외)의 원본과 정정본을 서로 정답으로 인정."""
    import re

    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT doc_id, title FROM documents WHERE source_type='disclosure' AND in_kb=1").fetchall()
    groups: dict[str, set[str]] = defaultdict(set)
    for d, t in rows:
        groups[re.sub(r"\[정정\]\s*", "", t).strip()].add(d)
    return {d: g for g in groups.values() if len(g) > 1 for d in g}


def doc_ranking(scores: np.ndarray, chunk_doc: np.ndarray) -> list[tuple[str, float]]:
    top = np.argpartition(-scores, CHUNK_POOL)[:CHUNK_POOL]
    top = top[np.argsort(-scores[top])]
    seen, out = set(), []
    for i in top:
        d = chunk_doc[i]
        if d not in seen:
            seen.add(d)
            out.append((d, float(scores[i])))
        if len(out) >= RANK_DEPTH:
            break
    return out


def evaluate(config: str, model: str, items: list[dict], q_vecs: np.ndarray, siblings: dict, corrections: dict) -> dict:
    chunks = read_chunks(config)
    alias_path = ROOT / "store" / "chunks" / f"{config}_doc_alias.json"
    alias = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    mat = embed_texts([c["text"] for c in chunks], model)
    chunk_doc = np.array([c["doc_id"] for c in chunks])
    indexed_docs = set(chunk_doc)
    per_item, by_type = [], defaultdict(list)
    unans_top1 = []
    for it, qv in zip(items, q_vecs):
        ranking = doc_ranking(mat @ qv, chunk_doc)
        if it["type"] == "unanswerable":
            unans_top1.append(ranking[0][1])
            continue
        gold = set(it["gold_doc_ids"])
        strict = set(gold)
        for g in gold:
            strict |= set(alias.get(g, []))           # 청크 중복으로 내용이 다른 문서 쪽에 남은 경우
        accept = set(strict) | set(it.get("also_accept", []))
        for g in gold:
            accept |= siblings.get(g, set()) | corrections.get(g, set())
        rank = next((r for r, (d, _) in enumerate(ranking, 1) if d in accept), None)
        rank_strict = next((r for r, (d, _) in enumerate(ranking, 1) if d in strict), None)
        row = {"id": it["id"], "type": it["type"], "rank": rank, "rank_strict": rank_strict, "top1_score": ranking[0][1],
               "missing_gold": sorted(g for g in gold if g not in indexed_docs),
               "top3": [d for d, _ in ranking[:3]]}
        per_item.append(row)
        by_type[it["type"]].append(row)

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        out = {"n": n}
        for k in KS:
            out[f"hit@{k}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3) if n else None
        out["mrr"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3) if n else None
        out["hit@3_strict"] = round(sum(1 for r in rows if r["rank_strict"] and r["rank_strict"] <= 3) / n, 3) if n else None
        return out

    return {
        "config": config,
        "model": model,
        "chunks": len(chunks),
        "overall": summarize(per_item),
        "by_type": {t: summarize(r) for t, r in sorted(by_type.items())},
        "answerable_top1_score_median": round(float(np.median([r["top1_score"] for r in per_item])), 4),
        "unanswerable_top1_score_median": round(float(np.median(unans_top1)), 4) if unans_top1 else None,
        "items": per_item,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    items = load_eval()
    q_vecs = embed_texts([it["q"] for it in items], args.model)
    siblings = event_siblings()
    corrections = correction_aliases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"retrieval_{args.model}.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for cfg in args.configs:
        results[cfg] = evaluate(cfg, args.model, items, q_vecs, siblings, corrections)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    types = sorted({t for r in results.values() for t in r["by_type"]})
    print(f"model={args.model}")
    print("| 설정 | 청크 | Hit@1 | Hit@3 | Hit@3(엄격) | Hit@5 | Hit@10 | MRR | " + " | ".join(f"{t} Hit@3" for t in types) + " |")
    print("|---|---|---|---|---|---|---|---|" + "---|" * len(types))
    for cfg, r in results.items():
        o = r["overall"]
        print(f"| {cfg} | {r['chunks']:,} | {o['hit@1']} | {o['hit@3']} | {o['hit@3_strict']} | {o['hit@5']} | {o['hit@10']} | {o['mrr']} | "
              + " | ".join(str(r["by_type"].get(t, {}).get("hit@3")) for t in types) + " |")
    for cfg, r in results.items():
        miss = [x["id"] for x in r["items"] if not x["rank"] or x["rank"] > 3]
        print(f"{cfg}: Hit@3 실패 {len(miss)}건 {miss}  | top1 유사도 중앙값 답변가능 {r['answerable_top1_score_median']} / 불가 {r['unanswerable_top1_score_median']}")


if __name__ == "__main__":
    main()
