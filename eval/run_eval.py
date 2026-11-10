"""평가 하네스: 평가셋 전체를 **챗봇과 같은 체인(rag.chain.ask)** 으로 돌려 검색·답변 품질을 측정한다.

    eval/eval_set.jsonl ─▶ rag.chain.ask(q) ─▶ 검색 지표 · 인용 지표 · 거절 지표 ─▶ LLM 채점(정확도·근거 충실도)
                                                                            └▶ 사람 검토용 CSV

지표 (문항 유형: fact·comparison·temporal·synthesis / unanswerable)
  검색      Hit@1·Hit@3·Hit@K(체인이 넘긴 문서 K개 안), MRR — 정답 확장 규칙은 eval/retrieval_eval.py와 같다
  인용      인용 정답률   답변이 인용한 문서 중 정답 문서가 있는 비율 (답변 가능 문항)
            잘못된 번호   [근거]에 없는 번호를 인용한 문항 수
  거절      거절 정확도   답변 불가 문항을 '[답변 불가]'로 거절한 비율
            과잉 거절     답변 가능 문항을 거절한 비율
  답변      정확도        LLM 채점 correct=1 · partial=0.5 · wrong=0 (answer_key 기준, 거절하면 wrong)
            근거 충실도   답변의 주장이 [근거]로 뒷받침되는 비율 (supported=1 · partial=0.5 · unsupported=0)
  비용      문항당 지연 시간, 입력·출력 토큰

LLM 채점은 참고값이다. 사람 검토용 CSV(manual_score 칸)로 직접 채점해 보정한다(docs/indexing.md §7 측정 지표).

실행:
    python -m eval.run_eval                          # 전체 79문항 (생성 + 채점)
    python -m eval.run_eval --ids F01 U01 --no-judge # 일부 문항, 채점 생략
    python -m eval.run_eval --types fact temporal --limit 10
    python -m eval.run_eval --chat-model gpt-5.4     # 생성 모델 비교
    python -m eval.run_eval --chunk-config fixed     # 청킹 설정 비교 (해당 컬렉션을 먼저 적재)
결과: eval/results/answer_eval_{tag}.json (요약 + 문항별), eval/results/answer_review_{tag}.csv
      tag 기본값 = {청킹 설정}__{생성 모델}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from eval.retrieval_eval import EVAL_SET, OUT_DIR, ROOT, correction_aliases, event_siblings, load_eval
from indexer.chunk import CONFIGS
from rag import chain, retriever

JUDGE_MODEL = retriever.ENV.get("JUDGE_MODEL") or "gpt-5.4"
TYPES = ("fact", "comparison", "temporal", "synthesis", "unanswerable")
CORRECT = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
GROUNDED = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


# --------------------------------------------------------------------------- #
# LLM 채점
# --------------------------------------------------------------------------- #
class Verdict(BaseModel):
    correctness: Literal["correct", "partial", "wrong"] = Field(
        description="정답 기준(answer_key)의 핵심 사실(상대방·금액·날짜·수치·결론)을 답변이 맞혔는가. "
                    "핵심을 모두 맞히면 correct, 일부만 맞히거나 중요한 조건이 빠지면 partial, 틀리거나 답하지 않으면 wrong")
    grounded: Literal["supported", "partial", "unsupported"] = Field(
        description="답변의 사실 주장이 [근거]에 실제로 있는가. 근거에 없는 수치·사실이 하나라도 핵심에 있으면 unsupported, "
                    "부수적인 주장 일부만 근거 밖이면 partial")
    conflict_handled: bool | None = Field(
        default=None, description="비교·종합 문항에서 출처별 차이·기준·시점을 함께 제시했는가 (그 외 유형은 null)")
    reason: str = Field(description="판정 이유 한두 문장 (한국어)")


JUDGE_SYSTEM = """너는 RAG 챗봇 답변을 채점하는 평가자다. 정답 기준(answer_key)과 챗봇이 받은 [근거]를 보고 판정한다.
- 정확도는 answer_key의 핵심 사실 기준으로 본다. 표현·어순 차이, 정답 기준에 없는 부가 정보는 감점하지 않는다.
- 단, 부가 정보가 정답 기준과 모순되면 감점한다. 수치는 단위·통화까지 확인한다.
- 근거 충실도는 [근거]만 보고 판단한다(네 배경지식으로 사실 여부를 판단하지 않는다).
- 비교·종합 문항은 출처마다 다른 수치를 '차이와 기준'으로 함께 제시해야 정답이다."""

JUDGE_HUMAN = """[문항 유형] {type}
[질문] {q}
[정답 기준] {answer_key}
[채점 참고] {note}

[챗봇이 받은 근거]
{context}

[챗봇 답변]
{answer}"""


def judge(item: dict, res: dict, hits_text: str, model: str) -> dict:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, api_key=retriever.ENV.get("OPENAI_API_KEY"), timeout=180, max_retries=3)
    v: Verdict = llm.with_structured_output(Verdict).invoke([
        ("system", JUDGE_SYSTEM),
        ("human", JUDGE_HUMAN.format(type=item["type"], q=item["q"], answer_key=item["answer_key"],
                                     note=item.get("note", "") or "-", context=hits_text, answer=res["answer"])),
    ])
    return v.model_dump()


# --------------------------------------------------------------------------- #
# 문항 하나 실행
# --------------------------------------------------------------------------- #
def accept_set(item: dict, alias: dict, siblings: dict, corrections: dict) -> set[str]:
    gold = set(item.get("gold_doc_ids") or [])
    acc = set(gold) | set(item.get("also_accept", []))
    for g in gold:
        acc |= set(alias.get(g, [])) | siblings.get(g, set()) | corrections.get(g, set())
    return acc


def run_item(item: dict, accept: set[str], chat_model: str, judge_model: str | None) -> dict:
    try:
        res = chain.ask(item["q"], model=chat_model)
    except Exception as e:  # 한 문항 실패가 전체를 멈추지 않도록
        return {"id": item["id"], "type": item["type"], "q": item["q"], "error": f"{type(e).__name__}: {e}"}
    doc_order = [s["doc_id"] for s in res["sources"]]
    rank = next((i for i, d in enumerate(doc_order, 1) if d in accept), None)
    cited_docs = [s["doc_id"] for s in res["sources"] if s["n"] in res["cited"]]
    raw_nums = {int(n) for n in chain.CITE.findall(res["answer"])}
    row = {
        "id": item["id"], "type": item["type"], "q": item["q"], "answer_key": item["answer_key"],
        "answer": res["answer"], "refused": res["refused"],
        "rank": rank, "retrieved": doc_order, "cited": cited_docs,
        "cited_gold": bool(set(cited_docs) & accept) if accept else None,
        "invalid_citations": sorted(raw_nums - {s["n"] for s in res["sources"]}),
        "latency_s": res["latency_s"],
        "input_tokens": res["usage"].get("input_tokens", 0), "output_tokens": res["usage"].get("output_tokens", 0),
        "sources": [{k: s[k] for k in ("n", "doc_id", "title", "published_at", "source_type", "score")} for s in res["sources"]],
    }
    if item["type"] == "unanswerable":
        row["correct"] = 1.0 if res["refused"] else 0.0
    elif res["refused"]:
        row.update(correct=0.0, judge={"correctness": "wrong", "grounded": None, "conflict_handled": None,
                                       "reason": "답변 가능 문항을 거절"})
    elif judge_model:
        try:       # 체인이 실제로 받은 근거로 채점한다 (재검색하면 LLM 재정렬처럼 비결정적인 검색에서 근거가 달라진다)
            v = judge(item, res, res["context"], judge_model)
            row.update(judge=v, correct=CORRECT[v["correctness"]], grounded=GROUNDED[v["grounded"]])
        except Exception as e:
            row["judge_error"] = f"{type(e).__name__}: {e}"
    return row


# --------------------------------------------------------------------------- #
# 집계
# --------------------------------------------------------------------------- #
def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows: list[dict], k_docs: int) -> dict:
    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["type"] != "unanswerable"]
    una = [r for r in ok if r["type"] == "unanswerable"]

    def block(rs: list[dict]) -> dict:
        a = [r for r in rs if r["type"] != "unanswerable"]
        u = [r for r in rs if r["type"] == "unanswerable"]
        out = {"n": len(rs)}
        if a:
            out.update({
                "hit@1": _mean([1.0 if r["rank"] and r["rank"] <= 1 else 0.0 for r in a]),
                "hit@3": _mean([1.0 if r["rank"] and r["rank"] <= 3 else 0.0 for r in a]),
                f"hit@{k_docs}": _mean([1.0 if r["rank"] else 0.0 for r in a]),
                "mrr": _mean([1 / r["rank"] if r["rank"] else 0.0 for r in a]),
                "accuracy": _mean([r.get("correct") for r in a]),
                "grounded": _mean([r.get("grounded") for r in a]),
                "cited_gold": _mean([1.0 if r["cited_gold"] else 0.0 for r in a]),
                "over_refusal": _mean([1.0 if r["refused"] else 0.0 for r in a]),
            })
            ch = [r["judge"]["conflict_handled"] for r in a if r.get("judge", {}).get("conflict_handled") is not None]
            if ch:
                out["conflict_handled"] = _mean([1.0 if x else 0.0 for x in ch])
        if u:
            out["refusal_accuracy"] = _mean([r["correct"] for r in u])
        return out

    by_type = defaultdict(list)
    for r in ok:
        by_type[r["type"]].append(r)
    return {
        "n": len(rows), "errors": [r["id"] for r in rows if "error" in r],
        "judge_errors": [r["id"] for r in ok if "judge_error" in r],
        "answerable": block(ans) if ans else {}, "unanswerable": block(una) if una else {},
        "by_type": {t: block(by_type[t]) for t in TYPES if by_type[t]},
        "invalid_citation_items": [r["id"] for r in ok if r["invalid_citations"]],
        "latency_s_median": _mean(sorted(r["latency_s"] for r in ok)[len(ok) // 2: len(ok) // 2 + 1]) if ok else None,
        "tokens_total": {"input": sum(r["input_tokens"] for r in ok), "output": sum(r["output_tokens"] for r in ok)},
    }


def print_report(s: dict, k: int) -> None:
    cols = ["n", "hit@1", "hit@3", f"hit@{k}", "mrr", "cited_gold", "accuracy", "grounded", "over_refusal", "refusal_accuracy"]
    print("| 구분 | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    rows = [("답변 가능", s["answerable"]), ("답변 불가", s["unanswerable"])] + list(s["by_type"].items())
    for name, b in rows:
        if b:
            print(f"| {name} | " + " | ".join("" if b.get(c) is None else str(b.get(c)) for c in cols) + " |")
    print(f"오류 {s['errors']} · 채점 오류 {s['judge_errors']} · 잘못된 인용 번호 {s['invalid_citation_items']}")
    print(f"지연 중앙값 {s['latency_s_median']}s · 토큰 {s['tokens_total']}")


def write_review_csv(rows: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "q", "answer_key", "answer", "refused", "retrieval_rank", "cited",
                    "llm_correctness", "llm_grounded", "llm_reason", "manual_score", "manual_note"])
        for r in rows:
            j = r.get("judge") or {}
            w.writerow([r["id"], r["type"], r["q"], r.get("answer_key", ""), r.get("answer", r.get("error", "")),
                        r.get("refused"), r.get("rank"), "; ".join(r.get("cited", [])),
                        j.get("correctness", "refused-ok" if r.get("type") == "unanswerable" and r.get("correct") else ""),
                        j.get("grounded", ""), j.get("reason", ""), "", ""])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--types", nargs="+", choices=TYPES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chat-model", default=chain.CHAT_MODEL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk-config", choices=CONFIGS, default=None,
                    help="검색할 컬렉션의 청킹 설정 (기본: 챗봇과 같은 structure). 먼저 `python -m indexer load --configs ...`")
    ap.add_argument("--retrieval", choices=retriever.MODES, default=None,
                    help="검색 방식 (기본: RETRIEVAL_MODE). dense · hybrid(+BM25) · hybrid_time(+질문 연도) · hybrid_rerank(+LLM 재정렬)")
    ap.add_argument("--embed-model", default=None,
                    help="질의·컬렉션 임베딩 모델 (기본: EMBED_MODEL). 먼저 `python -m indexer load --model ...`로 컬렉션 적재")
    ap.add_argument("--tag", help="결과 파일 이름 (기본: {청킹 설정}__{검색 방식}__{생성 모델}, 부분 실행이면 _partial)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    items = load_eval()
    if args.ids:
        items = [it for it in items if it["id"] in set(args.ids)]
    if args.types:
        items = [it for it in items if it["type"] in set(args.types)]
    if args.limit:
        items = items[:args.limit]
    partial = len(items) < len(load_eval())

    if args.chunk_config or args.retrieval or args.embed_model:
        retriever.configure(chunk_config=args.chunk_config, embed_model=args.embed_model, mode=args.retrieval)
    alias_path = ROOT / "store" / "chunks" / f"{retriever.CHUNK_CONFIG}_doc_alias.json"
    alias = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    siblings, corrections = event_siblings(), correction_aliases()
    retriever.vectorstore()                                  # 스레드 시작 전에 한 번 연다
    judge_model = None if args.no_judge else args.judge_model

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_item, it, accept_set(it, alias, siblings, corrections), args.chat_model, judge_model)
                for it in items]
        rows = []
        for i, f in enumerate(futs, 1):
            rows.append(f.result())
            r = rows[-1]
            print(f"[{i}/{len(items)}] {r['id']} rank={r.get('rank')} refused={r.get('refused')} "
                  f"correct={r.get('correct')} {r.get('error', '')}", flush=True)

    k_docs = retriever.K_DOCS
    summary = summarize(rows, k_docs)
    tag = args.tag or (f"{retriever.CHUNK_CONFIG}__{retriever.RETRIEVAL_MODE}__{args.chat_model}"
                       + ("_partial" if partial else ""))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run": {"chat_model": args.chat_model, "judge_model": judge_model, "embed_model": retriever.EMBED_MODEL,
                "collection": retriever.COLLECTION, "retrieval_mode": retriever.RETRIEVAL_MODE, "k_docs": k_docs, "max_chunks_per_doc": retriever.MAX_CHUNKS_PER_DOC,
                "eval_set": EVAL_SET.name, "n_items": len(items), "elapsed_s": round(time.perf_counter() - t0, 1),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "summary": summary,
        "items": rows,
    }
    (OUT_DIR / f"answer_eval_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_csv(rows, OUT_DIR / f"answer_review_{tag}.csv")
    print_report(summary, k_docs)
    print(f"→ eval/results/answer_eval_{tag}.json, answer_review_{tag}.csv")


if __name__ == "__main__":
    main()
