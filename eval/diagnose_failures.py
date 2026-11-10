"""실패 원인 계층 진단 — 평가 결과(answer_eval_*.json)의 실패 문항마다 원인 계층을 증거로 판정한다.

    python -m eval.diagnose_failures                                    # 현재 기본 실행 결과
    python -m eval.diagnose_failures --tag structure__hybrid_time__gpt-5.4-mini_prompt2

개선 루프: 측정 → 실패 사례 확인 → 원인 계층 특정 → 단일 변수 변경 → 재측정.
이 스크립트는 '원인 계층 특정' 단계다. 증상 → 계층 규칙 (판정 순서대로, 먼저 걸린 것 하나가 주 원인):

| 증상 | 계층 | 판정 증거 |
|---|---|---|
| 정답 문서가 지식베이스에 없음 | 데이터 | 정답(별칭 포함) 문서가 docs.db에 없음(수집) 또는 in_kb=0(선별) |
| 정답 문서가 있으나 검색되지 않음 | 인덱싱 | 지식베이스·Chroma에 있으나 넓은 검색(상위 60문서)에도 없음 |
| 검색은 되었으나 순위가 낮음 | 검색 | 넓은 검색 60위 안에는 있으나 상위 3 밖 (4~6위 포함) |
| 근거는 정확하나 답변이 부정확 | 생성 | 정답 문서가 1~3위인데 채점이 correct가 아님 |
| 근거 없는 내용을 생성 | 생성 | 근거 충실도 < 1 (주 원인과 별도로 표시) |
| 검색 상위가 동일 내용으로 채워짐 | 전처리 | 상위 6문서 중 같은 내용(같은 사건·정정 전후·본문 5-gram 유사도 ≥ 0.5)이 2쌍 이상 (보조 표시) |

답변 불가 문항의 실패(거절 안 함)는 '생성'(거절 규칙 미준수)으로, 평가셋 문항 오류가 확인된 것은 '평가셋'으로 둔다.
검색 재현에 질의 임베딩만 쓴다(문항당 1회). 생성·채점은 다시 하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import correction_aliases, event_siblings, load_eval  # noqa: E402
from eval.run_eval import accept_set  # noqa: E402
from rag import retriever  # noqa: E402

RESULTS = ROOT / "eval" / "results"
DOCS_DB = ROOT / "store" / "docs.db"
DEFAULT_TAG = "structure__hybrid_time__gpt-5.4-mini_prompt2"
WIDE_DOCS = 60            # '검색되지 않음'과 '순위가 낮음'을 가르는 넓은 검색 깊이
DUP_SIM = 0.5             # 본문 5-gram Jaccard (collector/dedup_news.py 재배포본 기준과 같음)
EVAL_SET_ERRORS = {"U10": "지식베이스 보강 후 근거가 생김 — 답변 불가 분류 오류 (failure_analysis §2.3)"}


def _grams(text: str) -> set[int]:
    s = re.sub(r"\s+|[^\w]", "", text or "")[:3000]
    return {hash(s[i:i + 5]) for i in range(max(0, len(s) - 4))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def chroma_doc_ids() -> set[str]:
    col = retriever.vectorstore()._collection
    ids, off, n = set(), 0, col.count()
    while off < n:
        ids |= {m.get("doc_id") for m in col.get(include=["metadatas"], limit=5000, offset=off)["metadatas"]}
        off += 5000
    return ids


def duplicate_pairs(doc_ids: list[str], con: sqlite3.Connection, siblings: dict, corrections: dict) -> list[tuple]:
    """상위 문서 중 같은 내용으로 볼 쌍: 같은 사건(event_id) · 정정 전후 공시 · 본문 유사도 ≥ DUP_SIM."""
    bodies = {d: _grams(b) for d, b in con.execute(
        f"SELECT doc_id, body FROM documents WHERE doc_id IN ({','.join('?' * len(doc_ids))})", doc_ids)}
    pairs = []
    for i, a in enumerate(doc_ids):
        for b in doc_ids[i + 1:]:
            why = ("같은 사건" if b in siblings.get(a, set()) else
                   "정정 전후" if b in corrections.get(a, set()) else None)
            sim = _jaccard(bodies.get(a, set()), bodies.get(b, set()))
            if why or sim >= DUP_SIM:
                pairs.append((a, b, why or f"본문 유사도 {sim:.2f}"))
    return pairs


def diagnose(tag: str) -> dict:
    run = json.loads((RESULTS / f"answer_eval_{tag}.json").read_text(encoding="utf-8"))
    retriever.configure(chunk_config=run["run"]["collection"].split("__")[0],
                        mode=run["run"].get("retrieval_mode") or "dense")
    items = {it["id"]: it for it in run["items"]}
    evalset = {it["id"]: it for it in load_eval()}
    alias_path = ROOT / "store" / "chunks" / f"{retriever.CHUNK_CONFIG}_doc_alias.json"
    alias = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    siblings, corrections = event_siblings(), correction_aliases()
    in_chroma = chroma_doc_ids()
    con = sqlite3.connect(f"file:{DOCS_DB.as_posix()}?mode=ro", uri=True)
    kb_flag = dict(con.execute("SELECT doc_id, in_kb FROM documents"))
    wide_mode = "hybrid_time" if retriever.RETRIEVAL_MODE == "hybrid_rerank" else retriever.RETRIEVAL_MODE

    rows = []
    for qid, it in items.items():
        ev = evalset[qid]
        answerable = it["type"] != "unanswerable"
        grade = ("과잉 거절" if it["refused"] else (it.get("judge") or {}).get("correctness")) if answerable else None
        rank = it.get("rank")
        dups = duplicate_pairs(it["retrieved"], con, siblings, corrections) if it.get("retrieved") else []
        row = {"id": qid, "type": it["type"], "q": it["q"], "grade": grade, "rank": rank,
               "grounded": it.get("grounded"), "dup_pairs": len(dups), "dup_detail": dups[:4]}

        if not answerable:
            if it.get("correct") == 1:
                continue
            row.update(layer="평가셋" if qid in EVAL_SET_ERRORS else "생성",
                       symptom="답변 불가 문항을 거절하지 않음",
                       evidence=EVAL_SET_ERRORS.get(qid, "거절 규칙 미준수 (거절 표시가 맨 앞에 없음)"))
            rows.append(row)
            continue
        failed = grade != "correct" or rank is None or rank > 3
        if not failed:
            continue

        acc = accept_set(ev, alias, siblings, corrections)
        gold = set(ev.get("gold_doc_ids") or [])
        in_db = [d for d in acc if d in kb_flag]
        in_kb = [d for d in in_db if kb_flag[d] == 1]
        in_idx = [d for d in in_kb if d in in_chroma]
        wide = retriever.search(it["q"], k_docs=WIDE_DOCS, pool=WIDE_DOCS * 10, mode=wide_mode)
        order = list(dict.fromkeys(h.doc_id for h in wide))
        wide_rank = next((i for i, d in enumerate(order, 1) if d in acc), None)
        row.update(gold=sorted(gold), wide_rank=wide_rank,
                   gold_in_db=len(in_db), gold_in_kb=len(in_kb), gold_in_index=len(in_idx))

        if not in_db:
            row.update(layer="데이터", symptom="정답 문서 미수집", evidence="정답·별칭 문서가 docs.db에 없음")
        elif not in_kb:
            row.update(layer="데이터", symptom="정답 문서가 지식베이스 선별에서 빠짐",
                       evidence=f"docs.db에는 있으나 in_kb=0: {in_db[:2]}")
        elif not in_idx:
            row.update(layer="인덱싱", symptom="정답 문서가 인덱스에 없음",
                       evidence="in_kb=1이나 Chroma에 청크가 없음 (청크 중복 제거로 빠짐)")
        elif wide_rank is None:
            row.update(layer="인덱싱", symptom="정답 문서가 있으나 검색되지 않음",
                       evidence=f"인덱스에 있으나 넓은 검색 상위 {WIDE_DOCS}문서에도 없음")
        elif rank is None or rank > 3:
            row.update(layer="검색", symptom="검색은 되었으나 순위가 낮음",
                       evidence=f"답변 근거 순위 {rank or '6위 밖'} · 넓은 검색 {wide_rank}위")
        else:
            row.update(layer="생성", symptom="근거는 정확하나 답변이 부정확",
                       evidence=f"정답 문서 {rank}위인데 채점 {grade}")
        rows.append(row)
    con.close()

    ungrounded = [it["id"] for it in items.values()
                  if it["type"] != "unanswerable" and it.get("grounded") is not None and it["grounded"] < 1]
    dup_heavy = sorted((r for r in rows if r["dup_pairs"] >= 2), key=lambda r: -r["dup_pairs"])
    all_dup = Counter()
    con = sqlite3.connect(f"file:{DOCS_DB.as_posix()}?mode=ro", uri=True)
    for it in items.values():
        if it.get("retrieved"):
            all_dup[min(len(duplicate_pairs(it["retrieved"], con, siblings, corrections)), 3)] += 1
    con.close()
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r["id"])
    return {"tag": tag, "run": run["run"], "wide_docs": WIDE_DOCS, "n_failed": len(rows),
            "by_layer": {k: v for k, v in sorted(by_layer.items(), key=lambda kv: -len(kv[1]))},
            "ungrounded": ungrounded, "dup_pairs_hist_all_items": {str(k): v for k, v in sorted(all_dup.items())},
            "dup_heavy_failed": [r["id"] for r in dup_heavy], "items": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="실패 원인 계층 진단")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="eval/results/answer_eval_{tag}.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rep = diagnose(args.tag)
    out = RESULTS / f"failure_layers__{args.tag}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.tag}] 실패 {rep['n_failed']}건 (답변 가능: Hit@3 실패 또는 correct 아님 · 답변 불가: 거절 실패)")
    for layer, ids in rep["by_layer"].items():
        print(f"  {layer:4s} {len(ids):2d}  {' '.join(ids)}")
    print(f"  근거 없는 내용(근거 충실도<1) {len(rep['ungrounded'])}건: {' '.join(rep['ungrounded'])}")
    print(f"  상위 6문서 중 같은 내용 쌍 수 분포(전 문항): {rep['dup_pairs_hist_all_items']} (3은 3쌍 이상)")
    print(f"  같은 내용 2쌍 이상인 실패 문항: {' '.join(rep['dup_heavy_failed']) or '없음'}")
    print(f"→ {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
