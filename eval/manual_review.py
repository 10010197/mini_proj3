"""사람 채점 — 블라인드 표본 검토표 생성과 LLM 채점 일치율 계산.

    python -m eval.manual_review sample      # 검토표 생성: eval/results/manual_review__{태그}.csv (기존 파일은 덮어쓰지 않음)
    python -m eval.manual_review agree       # 채운 검토표로 사람 정확도·LLM 일치율·카파 계산

- 표본: 답변 가능 문항에서 유형별 비례 층화 30문항(사실 14 · 비교 5 · 시점 6 · 종합 5), 고정 시드.
- 블라인드: 검토표에는 LLM 채점(correct/partial/wrong)과 채점 이유를 넣지 않는다. 사람이 먼저 채점하고, 비교는 agree 단계에서 한다.
- 채점 기준 (사람 채점자 확정, 2026-11-10): manual_score 칸에 1 · 0.5 · 0.
  1   = 질문이 묻는 내용을 정확히 답함. 질문이 묻지 않은 세부 조건(정답 기준에만 있는 비율·기간·순위 등)이 빠져도 1.
  0.5 = 질문이 묻는 내용 중 일부만 답함 (예: 비교 문항에서 한쪽만, '대상과 금액' 중 하나만).
  0   = 질문이 묻는 핵심이 틀림, 또는 근거가 있는데 거절함.
  LLM 채점(gpt-5.4)은 정답 기준의 부가 조건이 빠지면 partial을 준다. 그래서 사람 점수가 더 높게 나오는 차이는
  대부분 이 기준 차이로 예상되며, agree 결과에서 '사람이 더 높게' 문항으로 확인한다.
- 결과: eval/results/manual_agreement__{태그}.json — 제출 점검(eval.check_submission)이 이 파일을 근거로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
DEFAULT_TAG = "structure__hybrid_time__gpt-5.4-mini_prompt2"
QUOTA = {"fact": 14, "comparison": 5, "temporal": 6, "synthesis": 5}
SEED = 20261110
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def sheet_path(tag: str) -> Path:
    return RESULTS / f"manual_review__{tag}.csv"


def sample(tag: str) -> Path:
    out = sheet_path(tag)
    if out.exists():
        sys.exit(f"{out.relative_to(ROOT)} 가 이미 있습니다. 채점한 내용을 지키기 위해 덮어쓰지 않습니다.")
    run = json.loads((RESULTS / f"answer_eval_{tag}.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    rows = []
    for t, k in QUOTA.items():
        pool = sorted([it for it in run["items"] if it["type"] == t], key=lambda x: x["id"])
        for it in sorted(rng.sample(pool, k), key=lambda x: x["id"]):
            srcs = " | ".join(f"[{s['n']}] {s['title']} ({s['published_at']})" for s in it.get("sources", []))
            rows.append({"id": it["id"], "type": t, "q": it["q"], "answer_key": it["answer_key"],
                         "answer": it["answer"], "sources": srcs, "manual_score": "", "manual_note": ""})
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def kappa(a: list, b: list) -> float | None:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def agree(tag: str) -> dict:
    df = pd.read_csv(sheet_path(tag), encoding="utf-8-sig")
    df = df[pd.to_numeric(df["manual_score"], errors="coerce").isin([0, 0.5, 1])].copy()
    df["manual_score"] = df["manual_score"].astype(float)
    run = {it["id"]: it for it in json.loads((RESULTS / f"answer_eval_{tag}.json").read_text(encoding="utf-8"))["items"]}
    df["llm_score"] = [0.0 if run[i]["refused"] else SCORE.get((run[i].get("judge") or {}).get("correctness"), 0.0)
                       for i in df["id"]]
    m, l = df["manual_score"].tolist(), df["llm_score"].tolist()
    conf = pd.crosstab(df["llm_score"], df["manual_score"]).to_dict()   # to_dict: {열(사람): {행(LLM): 수}}
    res = {"tag": tag, "n_scored": len(df), "n_sheet": len(pd.read_csv(sheet_path(tag), encoding="utf-8-sig")),
           "manual_accuracy": round(sum(m) / len(m), 3) if m else None,
           "llm_accuracy_same_items": round(sum(l) / len(l), 3) if l else None,
           "exact_agreement": round(sum(x == y for x, y in zip(m, l)) / len(m), 3) if m else None,
           "within_half_point": round(sum(abs(x - y) <= 0.5 for x, y in zip(m, l)) / len(m), 3) if m else None,
           "cohen_kappa": kappa(m, l),
           "manual_higher": int(sum(x > y for x, y in zip(m, l))), "manual_lower": int(sum(x < y for x, y in zip(m, l))),
           "by_type": {t: {"n": int(len(g)), "manual": round(g["manual_score"].mean(), 3), "llm": round(g["llm_score"].mean(), 3)}
                       for t, g in df.groupby("type")},
           "confusion_manual_to_llm": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in conf.items()}}
    (RESULTS / f"manual_agreement__{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="사람 채점 표본과 LLM 채점 일치율")
    ap.add_argument("action", choices=["sample", "agree"])
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.action == "sample":
        p = sample(args.tag)
        print(f"검토표 {sum(QUOTA.values())}문항 → {p.relative_to(ROOT).as_posix()}\n"
              "manual_score에 1 / 0.5 / 0을 적은 뒤: python -m eval.manual_review agree")
    else:
        r = agree(args.tag)
        print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
