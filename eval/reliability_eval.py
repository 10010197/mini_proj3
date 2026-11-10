"""답변별 신뢰성 지표 검증 — 챗봇이 보여주는 주의 표시가 실제로 정확도를 가르는지 평가 결과로 확인한다.

    python -m eval.reliability_eval

- 챗봇(app/chat.py)과 같은 rag.reliability.assess()·flags()를 평가 79문항의 저장된 결과에 적용한다(API 호출 없음).
- 후보 지표마다 '표시가 붙은 답변'과 '안 붙은 답변'의 정확도(LLM 채점 전체 · 사람 채점 표본)를 비교한다.
- 정확도를 가른 지표만 챗봇의 주의 표시로 쓴다. 가르지 못한 후보도 함께 기록한다(채택하지 않은 근거).
결과: eval/results/reliability_validation.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import reliability as R  # noqa: E402

RESULTS = ROOT / "eval" / "results"

# 검토한 후보 (채택: rag/reliability.py의 FLAG_RULES). 기각 사유 (2026-11-10 측정):
#   single_source — 오히려 정확도가 높다(단일 문서 사실 질문이 쉬움). secondary_only — LLM 0.63 vs 0.632로 차이 없음.
#   low_citation_coverage — 14건 중 11건이 채택 지표와 겹치고, 나머지 3건 정확도 0.67로 추가 정보가 적음.
#   year_mismatch — 2건뿐, 둘 다 정답.
CANDIDATES = {
    "multi_source (채택)": lambda m, ref: m["n_cited"] >= R.MULTI_SOURCE,
    "low_similarity (채택)": lambda m, ref: m["sim_max"] is not None and m["sim_max"] < ref["unanswerable_median"],
    "single_source (기각)": lambda m, ref: m["n_cited"] == 1,
    "secondary_only (기각)": lambda m, ref: m["primary_share"] == 0,
    "low_citation_coverage<0.5 (기각)": lambda m, ref: m["citation_coverage"] is not None and m["citation_coverage"] < 0.5,
    "year_mismatch (기각)": lambda m, ref: m["in_year_share"] == 0,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser(description="신뢰성 주의 표시 검증")
    ap.add_argument("--tag", default=None, help="평가 결과 태그 (기본: 챗봇 현재 설정 rag.reliability.REF_TAG)")
    args = ap.parse_args()
    current = R.REF_TAG
    if args.tag:
        R.REF_TAG = args.tag
        R.reference.cache_clear()
    ref = R.reference()
    run = json.loads((RESULTS / f"answer_eval_{R.REF_TAG}.json").read_text(encoding="utf-8"))
    human = {}
    sheet = RESULTS / f"manual_review__{R.REF_TAG}.csv"
    if sheet.exists():
        with sheet.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    human[r["id"]] = float(r["manual_score"])
                except (TypeError, ValueError):
                    pass
    rows = []
    for it in run["items"]:
        if it["type"] == "unanswerable" or it["refused"]:
            continue
        n_by = {s["doc_id"]: s["n"] for s in it["sources"]}
        m = R.assess(it["q"], it["answer"], it["sources"], [n_by[d] for d in it["cited"] if d in n_by])
        rows.append((m, it.get("correct"), human.get(it["id"]), it["id"]))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (round(sum(xs) / len(xs), 3) if xs else None), len(xs)

    out = {"tag": R.REF_TAG, "n_answered": len(rows), "unanswerable_median": ref["unanswerable_median"],
           "multi_source_threshold": R.MULTI_SOURCE, "candidates": {}}
    print(f"[{R.REF_TAG}] 거절하지 않은 답변 가능 {len(rows)}문항 · 사람 채점 {sum(1 for r in rows if r[2] is not None)}문항")
    print(f"{'후보':34s} {'표시':>4} | LLM 표시/없음 | 사람 표시(n)/없음(n)")
    for name, fn in CANDIDATES.items():
        on = [r for r in rows if fn(r[0], ref)]
        off = [r for r in rows if not fn(r[0], ref)]
        lo, lf = mean(r[1] for r in on)[0], mean(r[1] for r in off)[0]
        (ho, hn_on), (hf, hn_off) = mean(r[2] for r in on), mean(r[2] for r in off)
        out["candidates"][name] = {"n": len(on), "ids": [r[3] for r in on], "llm_on": lo, "llm_off": lf,
                                   "human_on": ho, "human_n_on": hn_on, "human_off": hf, "human_n_off": hn_off}
        print(f"{name:34s} {len(on):>4} | {lo} / {lf} | {ho}({hn_on}) / {hf}({hn_off})")
    out["no_flag"] = ref["validation"]["no_flag"]
    v = out["no_flag"]
    print(f"주의 표시 없음 {v['n']}건: LLM {v['llm']} · 사람 {v['human']} (n={v['human_n']})")
    paths = [RESULTS / f"reliability_validation__{R.REF_TAG}.json"]
    if R.REF_TAG == current:                    # 챗봇 현재 설정의 검증은 기본 파일 이름으로도 둔다 (제출 점검이 읽음)
        paths.append(RESULTS / "reliability_validation.json")
    for p in paths:
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {p.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
