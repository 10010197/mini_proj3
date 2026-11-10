"""대표 질문별 근거 커버리지 점검 (부족 자료 탐색).

eval/questions.json의 각 질문에 대해, 지식베이스(in_kb=1)와 전체 수집본에서
검색식(must: 동의어 그룹을 모두 만족, must_by_source: 출처별 덮어쓰기)에 걸리는 문서를 출처 종류별로 센다.
임베딩 전 단계라 키워드 검색으로 '근거가 존재하는가'만 본다.

판정
  OK       : 질문이 요구하는 출처(needs)마다 지식베이스에 MIN_DOCS건 이상
  SELECT   : 수집은 됐지만 지식베이스 선별에서 빠져 부족 → 선별 규칙 조정
  COLLECT  : 수집 자체가 부족 → 직접 수집으로 보강
  (policy 처럼 아예 없는 출처는 COLLECT)

실행:
    python -m eval.kb_coverage            # 표 출력 + eval/kb_coverage.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "store" / "docs.db"
QUESTIONS = ROOT / "eval" / "questions.json"
MIN_DOCS = {"news": 5, "disclosure": 3, "patent": 10, "report": 3, "policy": 2}


def matcher(groups: list[list[str]]):
    pats = [re.compile("|".join(re.escape(w) for w in g), re.I) for g in groups]
    return lambda text: all(p.search(text) for p in pats)


def run() -> list[dict]:
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    with sqlite3.connect(DB) as con:
        docs = pd.read_sql("SELECT doc_id, source_type, title, body, published_at, in_kb FROM documents", con)
    text = docs["title"] + "\n" + docs["body"]
    out = []
    for q in qs:
        if "must" not in q:
            out.append({"id": q["id"], "verdict": "N/A", "note": q.get("expect")})
            continue
        m = text.map(matcher(q["must"]))
        # 출처별 검색식 덮어쓰기 (예: 공시에는 정책 용어가 없으므로 다른 검색식으로 찾는다)
        for src, groups in q.get("must_by_source", {}).items():
            is_src = docs["source_type"] == src
            m = (m & ~is_src) | (is_src & text.map(matcher(groups)))
        if q.get("since"):
            m &= docs["published_at"] >= q["since"]
        if q.get("until"):
            m &= docs["published_at"] <= q["until"] + "T23:59:59"
        hit = docs[m]
        kb = hit[hit["in_kb"] == 1]["source_type"].value_counts().to_dict()
        allc = hit["source_type"].value_counts().to_dict()
        lacking = [s for s in q["needs"] if kb.get(s, 0) < MIN_DOCS[s]]
        if not lacking:
            verdict = "OK"
        elif all(allc.get(s, 0) >= MIN_DOCS[s] for s in lacking):
            verdict = "SELECT"
        else:
            verdict = "COLLECT"
        out.append({"id": q["id"], "type": q["type"], "needs": q["needs"], "kb": kb, "all": allc,
                    "lacking": lacking, "verdict": verdict})
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    (ROOT / "eval" / "kb_coverage.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    src = ["news", "disclosure", "patent", "report", "policy"]
    print("| ID | 판정 | " + " | ".join(src) + " | 부족 |")
    print("|---|---|" + "---|" * len(src) + "---|")
    for r in res:
        if r["verdict"] == "N/A":
            print(f"| {r['id']} | 거절 문항 ({r['note']}) |" + " |" * len(src) + " |")
            continue
        cells = [f"{r['kb'].get(s, 0)} / {r['all'].get(s, 0)}" + ("*" if s in r["needs"] else "") for s in src]
        print(f"| {r['id']} | {r['verdict']} | " + " | ".join(cells) + f" | {', '.join(r['lacking'])} |")
    print("\n셀 = 지식베이스 / 전체 수집, * = 질문이 요구하는 출처")


if __name__ == "__main__":
    main()
