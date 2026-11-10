"""검색 단일 변수 스윕 — 생성·채점 없이 체인과 같은 retriever.search()의 순위만 측정한다.

    python -m eval.sweep_retrieval --param pool --values 40 80 160 400
    python -m eval.sweep_retrieval --param max_chunks_per_doc --values 1 2 3

- 변수 하나만 바꾸고 나머지(청킹·임베딩·검색 방식·k_docs·프롬프트)는 현재 기본값으로 고정한다.
- 질의 임베딩은 문항당 한 번만 만들어 재사용한다(값마다 같은 벡터 → 순위 차이는 변수 효과만).
- 정답 판정은 run_eval과 같다(accept_set: 별칭·같은 사건·정정 전후·also_accept).
- 결과: eval/results/retrieval_sweep__{param}.json — 여기서 고른 값만 run_eval로 생성·채점까지 재측정한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import correction_aliases, event_siblings, load_eval  # noqa: E402
from eval.run_eval import accept_set  # noqa: E402
from rag import retriever  # noqa: E402

PARAMS = {"pool": int, "k_docs": int, "max_chunks_per_doc": int}


def measure(items: list[dict], accepts: dict, kw: dict) -> dict:
    rows = []
    for it in items:
        docs = list(dict.fromkeys(h.doc_id for h in retriever.search(it["q"], **kw)))
        rank = next((i for i, d in enumerate(docs, 1) if d in accepts[it["id"]]), None)
        rows.append({"id": it["id"], "type": it["type"], "rank": rank})

    def block(rs):
        n = len(rs)
        hit = lambda k: round(sum(1 for r in rs if r["rank"] and r["rank"] <= k) / n, 3)  # noqa: E731
        return {"n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@6": hit(6),
                "mrr": round(sum(1 / r["rank"] for r in rs if r["rank"]) / n, 3)}
    by_type = {t: block([r for r in rows if r["type"] == t]) for t in ("fact", "comparison", "temporal", "synthesis")}
    return {"overall": block(rows), "by_type": by_type, "ranks": {r["id"]: r["rank"] for r in rows}}


def main() -> None:
    ap = argparse.ArgumentParser(description="검색 단일 변수 스윕")
    ap.add_argument("--param", choices=list(PARAMS), required=True)
    ap.add_argument("--values", nargs="+", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    vs = retriever.vectorstore()
    emb = vs.embeddings
    cached = lru_cache(maxsize=None)(emb.embed_query)
    object.__setattr__(emb, "embed_query", cached) if hasattr(emb, "__dict__") else None

    items = [it for it in load_eval() if it["type"] != "unanswerable"]
    alias_path = ROOT / "store" / "chunks" / f"{retriever.CHUNK_CONFIG}_doc_alias.json"
    alias = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    siblings, corrections = event_siblings(), correction_aliases()
    accepts = {it["id"]: accept_set(it, alias, siblings, corrections) for it in items}

    base = {"pool": retriever.POOL, "k_docs": retriever.K_DOCS, "max_chunks_per_doc": retriever.MAX_CHUNKS_PER_DOC}
    out = {"param": args.param, "fixed": {**base, "mode": retriever.RETRIEVAL_MODE, "chunk_config": retriever.CHUNK_CONFIG,
                                          "embed_model": retriever.EMBED_MODEL}, "runs": {}}
    print(f"고정: {out['fixed']}  변수: {args.param} ∈ {args.values}  (답변 가능 {len(items)}문항)")
    print(f"{'값':>6} | Hit@1  Hit@3  Hit@6  MRR   | Hit@3 사실·비교·시점·종합 | 초")
    for v in args.values:
        kw = {**base, args.param: PARAMS[args.param](v)}
        t0 = time.perf_counter()
        res = measure(items, accepts, kw)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        out["runs"][v] = res
        o, bt = res["overall"], res["by_type"]
        print(f"{v:>6} | {o['hit@1']:.3f}  {o['hit@3']:.3f}  {o['hit@6']:.3f}  {o['mrr']:.3f} | "
              f"{bt['fact']['hit@3']:.3f} {bt['comparison']['hit@3']:.3f} {bt['temporal']['hit@3']:.3f} "
              f"{bt['synthesis']['hit@3']:.3f} | {res['seconds']}")
    first = args.values[0]
    for v in args.values[1:]:
        a, b = out["runs"][first]["ranks"], out["runs"][v]["ranks"]
        gain = [q for q in a if (b[q] or 99) <= 3 < (a[q] or 99)]
        loss = [q for q in a if (a[q] or 99) <= 3 < (b[q] or 99)]
        print(f"  {first} → {v}: Hit@3 새로 맞힘 {gain} · 잃음 {loss}")
    path = ROOT / "eval" / "results" / f"retrieval_sweep__{args.param}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
