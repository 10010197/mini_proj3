"""④ 챗봇 계층 — 답변별 신뢰성 판단 근거 (수치).

체인 결과(근거·인용·답변)만으로 계산한다. LLM·임베딩 호출이 없고, 저장소에 쓰지 않는다.
합성 점수 하나로 뭉치지 않고, 판단에 쓰는 원자료 수치와 '주의 표시'를 따로 보여준다.
주의 표시의 기준값은 평가 결과(eval/results)에서 읽는다 — 기준이 임의값이 아니라 측정값에 묶인다.

    m = assess(question, answer, sources, cited)     # sources: chain.finalize()["sources"], cited: 근거 번호 목록
    ref = reference()                                # 평가 기준값 (없으면 {})

지표
- 근거: 받은 문서 수, 답변이 인용한 문서 수
- 출처 구성: 인용 문서의 출처 종류별 수와 1차 자료(공시·정부 보도자료) 비율
- 유사도: 인용 문서의 최고·평균 코사인 유사도, 평가에서 '정답 근거를 찾은 문항'의 최고 유사도 분포에서의 백분위
- 인용 범위: 답변의 사실 문장 중 [n] 인용이 붙은 비율
- 시점: 인용 근거의 발행일 범위, 질문에 연도가 있으면 그 연도의 근거 비율
- 출처 다양성: 인용 문서의 서로 다른 발행처 수
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rag import retriever

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
REF_TAG = "structure__3large__hybrid_time__gpt-5.4-mini_prompt2"     # 현재 기본 설정(임베딩 3-large)의 평가 결과
# 사람 채점은 교체 전 3-small 실행에만 있다 → 사람 정확도는 그 실행 기준으로 따로 읽는다
HUMAN_TAG = "structure__hybrid_time__gpt-5.4-mini_prompt2"
PRIMARY = {"disclosure", "policy"}                           # 1차 자료 (rag/prompts.py 신뢰도 안내와 같은 구분)
CITE = re.compile(r"\[(\d{1,2})\]")
SENT = re.compile(r"(?<=[.!?다요])\s+|\n+")


def _sentences(answer: str) -> list[str]:
    """사실 문장 후보: 제목·빈 줄·짧은 머리말·거절 문구 줄을 뺀 문장."""
    out = []
    for s in SENT.split(answer or ""):
        t = s.strip(" -*#>\t")
        if len(re.sub(r"\[\d+\]|\W", "", t)) < 8 or t.startswith("[답변 불가]") or t.endswith(":"):
            continue
        out.append(t)
    return out


def assess(question: str, answer: str, sources: list[dict], cited: list[int] | set[int]) -> dict:
    cited = set(cited or [])
    used = [s for s in sources if s["n"] in cited]
    scores = [s["score"] for s in used if s.get("score") is not None]
    types: dict[str, int] = {}
    for s in used:
        types[s["source_type"]] = types.get(s["source_type"], 0) + 1
    sents = _sentences(answer)
    with_cite = [x for x in sents if CITE.search(x)]
    dates = sorted(str(s.get("published_at") or "")[:10] for s in used if s.get("published_at"))
    years = retriever.query_years(question)
    in_year = [s for s in used if str(s.get("published_at") or "")[:4].isdigit() and int(str(s["published_at"])[:4]) in years]
    return {
        "n_sources": len(sources), "n_cited": len(used),
        "source_mix": types,
        "primary_share": round(sum(types.get(t, 0) for t in PRIMARY) / len(used), 3) if used else None,
        "sim_max": round(max(scores), 3) if scores else None,
        "sim_mean": round(sum(scores) / len(scores), 3) if scores else None,
        "n_sentences": len(sents), "cited_sentences": len(with_cite),
        "citation_coverage": round(len(with_cite) / len(sents), 3) if sents else None,
        "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None,
        "query_years": years, "in_year_share": round(len(in_year) / len(used), 3) if (used and years) else None,
        "publishers": len({s.get("publisher") for s in used if s.get("publisher")}),
    }


MULTI_SOURCE = 4      # 인용 문서 수 기준 — 평가에서 4건 이상을 인용한 답변의 정확도가 가장 크게 갈렸다 (2~5 비교)

FLAG_RULES = {
    "multi_source": "여러 문서를 종합한 답변",
    "low_similarity": "근거와 질문의 유사도가 낮음",
}


def _flag_keys(m: dict, unans_median: float | None) -> list[str]:
    keys = []
    if m["n_cited"] >= MULTI_SOURCE:
        keys.append("multi_source")
    if unans_median is not None and m["sim_max"] is not None and m["sim_max"] < unans_median:
        keys.append("low_similarity")
    return keys


@lru_cache(maxsize=1)
def reference() -> dict:
    """평가 기준값과 주의 표시 검증 — 현재 기본 설정의 평가 결과(및 사람 채점 표본)에서 계산한다.

    주의 표시는 평가에서 '표시가 붙은 답변'과 '안 붙은 답변'의 정확도를 실제로 갈랐던 것만 쓴다.
    (근거 1건뿐 · 2차 자료뿐 · 인용 문장 비율 · 질문 연도 불일치는 평가에서 정확도를 가르지 못해 표시하지 않는다.)
    """
    p = RESULTS / f"answer_eval_{REF_TAG}.json"
    if not p.exists():
        return {}
    run = json.loads(p.read_text(encoding="utf-8"))
    human = {}
    sheet = RESULTS / f"manual_review__{REF_TAG}.csv"
    if sheet.exists():
        import csv
        with sheet.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    human[r["id"]] = float(r["manual_score"])
                except (TypeError, ValueError):
                    pass
    found, unans = [], []
    for it in run["items"]:
        sc = [s["score"] for s in it.get("sources") or [] if s.get("score") is not None]
        if sc:
            (unans if it["type"] == "unanswerable" else found if it.get("rank") else []).append(max(sc))
    found.sort()
    unans.sort()
    um = unans[len(unans) // 2] if unans else None

    rows = []                                   # 거절하지 않은 답변 가능 문항
    for it in run["items"]:
        if it["type"] == "unanswerable" or it["refused"]:
            continue
        n_by = {s["doc_id"]: s["n"] for s in it["sources"]}
        m = assess(it["q"], it["answer"], it["sources"], [n_by[d] for d in it["cited"] if d in n_by])
        rows.append((_flag_keys(m, um), it.get("correct"), human.get(it["id"])))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (round(sum(xs) / len(xs), 3), len(xs)) if xs else (None, 0)

    validation = {}
    for k in FLAG_RULES:
        on = [r for r in rows if k in r[0]]
        off = [r for r in rows if k not in r[0]]
        validation[k] = {"n": len(on), "n_off": len(off),
                         "llm_on": mean(r[1] for r in on)[0], "llm_off": mean(r[1] for r in off)[0],
                         "human_on": mean(r[2] for r in on)[0], "human_n_on": mean(r[2] for r in on)[1],
                         "human_off": mean(r[2] for r in off)[0], "human_n_off": mean(r[2] for r in off)[1]}
    none = [r for r in rows if not r[0]]
    validation["no_flag"] = {"n": len(none), "llm": mean(r[1] for r in none)[0],
                             "human": mean(r[2] for r in none)[0], "human_n": mean(r[2] for r in none)[1]}
    a = run["summary"]["answerable"]
    ref = {"tag": REF_TAG, "found_top_sims": found, "found_median": found[len(found) // 2] if found else None,
           "unanswerable_median": um, "hit3": a["hit@3"], "grounded": a["grounded"], "llm_accuracy": a["accuracy"],
           "refusal_accuracy": run["summary"]["unanswerable"]["refusal_accuracy"],
           "n_answered": len(rows), "validation": validation}
    if len(human) >= 30:
        ref.update(human_accuracy=round(sum(human.values()) / len(human), 3), human_n=len(human), human_tag=REF_TAG)
    else:                                       # 현재 설정에 사람 채점이 없으면 교체 전 실행의 사람 채점을 '참고'로 표시
        m = RESULTS / f"manual_agreement__{HUMAN_TAG}.json"
        if m.exists():
            agr = json.loads(m.read_text(encoding="utf-8"))
            if agr.get("n_scored", 0) >= 30:
                ref.update(human_accuracy=agr["manual_accuracy"], human_n=agr["n_scored"], human_tag=HUMAN_TAG)
    return ref


def percentile(value: float | None, ref: dict) -> int | None:
    xs = ref.get("found_top_sims") or []
    if value is None or not xs:
        return None
    return round(100 * sum(1 for x in xs if x <= value) / len(xs))


def flags(m: dict, ref: dict) -> list[dict]:
    """주의 표시 — 평가에서 정확도를 가른 기준만, 그 근거 수치와 함께 돌려준다."""
    out = []
    for k in _flag_keys(m, ref.get("unanswerable_median")):
        v = ref.get("validation", {}).get(k, {})
        if k == "multi_source":
            why = f"인용 근거 {m['n_cited']}건 (기준 {MULTI_SOURCE}건 이상) — 비교·종합·경과형 답변"
        else:
            why = (f"최고 유사도 {m['sim_max']:.3f} — 평가의 답변 불가 문항 중앙값 "
                   f"{ref['unanswerable_median']:.3f}보다 낮음")
        ev = (f"평가에서 이 표시가 붙은 답변 {v.get('n')}건의 정확도 LLM {v.get('llm_on')} "
              f"(없는 답변 {v.get('llm_off')})")
        if v.get("human_n_on"):
            ev += f" · 사람 채점 {v['human_on']} (n={v['human_n_on']}, 없는 답변 {v['human_off']})"
        out.append({"key": k, "label": FLAG_RULES[k], "why": why, "evidence": ev})
    return out
