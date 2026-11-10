"""임베딩 선택의 정량적 근거 — 모델(3-small vs 3-large) × 차원(전체·1024·512·256) 검색 비교.

    python -m eval.embedding_comparison          # 두 모델 캐시가 모두 있어야 한다 (python -m indexer embed --model ...)

- 청킹은 structure로 고정, 벡터 검색 단독(코사인 전체 행렬 곱)으로 순위를 매긴다. 순위·정답 판정은 eval/retrieval_eval.py와 같다.
  → 모델·차원만 바뀌고 나머지는 같은 단일 변수 비교다.
- 차원 축소: text-embedding-3 계열은 앞쪽 차원만 잘라 다시 정규화해도 쓸 수 있게 학습됐다(API의 dimensions 옵션과 같은 방식).
  그래서 캐시 벡터를 잘라 추가 API 비용 없이 비교한다.
- 비용: 청크 텍스트의 실제 토큰 수(tiktoken cl100k_base) × 단가. 저장 용량: 청크 수 × 차원 × 4바이트(float32).
- --chain: 같은 비교를 챗봇 검색기(rag.retriever.search, Chroma)로 검색 방식별(dense·hybrid·hybrid_time) 다시 잰다.
  3-large 컬렉션이 적재돼 있어야 한다: python -m indexer load --configs structure --model text-embedding-3-large
결과: eval/results/embedding_comparison.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import KS, correction_aliases, doc_ranking, event_siblings, load_eval  # noqa: E402
from indexer.__main__ import read_chunks  # noqa: E402
from indexer.embed import embed_texts  # noqa: E402

CONFIG = "structure"
MODELS = {"text-embedding-3-small": {"dim": 1536, "usd_per_mtok": 0.02},
          "text-embedding-3-large": {"dim": 3072, "usd_per_mtok": 0.13}}
DIMS = {"text-embedding-3-small": [1536, 1024, 512, 256], "text-embedding-3-large": [3072, 1536, 1024, 512, 256]}
OUT = ROOT / "eval" / "results" / "embedding_comparison.json"


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def rank_all(mat: np.ndarray, q: np.ndarray, items: list[dict], chunk_doc: np.ndarray, accepts: dict) -> dict:
    rows, by_type = [], defaultdict(list)
    for it, qv in zip(items, q):
        ranking = doc_ranking(mat @ qv, chunk_doc)
        rank = next((r for r, (d, _) in enumerate(ranking, 1) if d in accepts[it["id"]]), None)
        row = {"id": it["id"], "type": it["type"], "rank": rank}
        rows.append(row)
        by_type[it["type"]].append(row)

    def summ(rs):
        n = len(rs)
        o = {f"hit@{k}": round(sum(1 for r in rs if r["rank"] and r["rank"] <= k) / n, 3) for k in KS}
        o["mrr"] = round(sum(1 / r["rank"] for r in rs if r["rank"]) / n, 3)
        return o
    return {"overall": summ(rows), "by_type": {t: summ(r) for t, r in sorted(by_type.items())},
            "ranks": {r["id"]: r["rank"] for r in rows}}


def chain_compare(items: list[dict], accepts: dict) -> dict:
    """챗봇과 같은 retriever.search()로 모델 × 검색 방식 순위 비교 (생성·채점 없음)."""
    from eval.sweep_retrieval import measure
    from rag import retriever

    out = {}
    for model in MODELS:
        for mode in ("dense", "hybrid", "hybrid_time"):
            retriever.configure(embed_model=model, mode=mode)
            kw = {"pool": retriever.POOL, "k_docs": retriever.K_DOCS, "max_chunks_per_doc": retriever.MAX_CHUNKS_PER_DOC}
            r = measure(items, accepts, kw)
            out[f"{model}|{mode}"] = r
            o, bt = r["overall"], r["by_type"]
            print(f"[체인] {model:24s} {mode:12s} | Hit@1 {o['hit@1']:.3f} Hit@3 {o['hit@3']:.3f} Hit@6 {o['hit@6']:.3f} MRR {o['mrr']:.3f} | "
                  f"Hit@3 사실 {bt['fact']['hit@3']} 비교 {bt['comparison']['hit@3']} 시점 {bt['temporal']['hit@3']} 종합 {bt['synthesis']['hit@3']}")
    retriever.configure(embed_model="text-embedding-3-small", mode="hybrid_time")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--chain" in sys.argv:
        items = [it for it in load_eval() if it["type"] != "unanswerable"]
        from eval.run_eval import accept_set
        alias_path = ROOT / "store" / "chunks" / f"{CONFIG}_doc_alias.json"
        alias = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
        sib, cor = event_siblings(), correction_aliases()
        accepts = {it["id"]: accept_set(it, alias, sib, cor) for it in items}
        data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        data["chain"] = chain_compare(items, accepts)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    chunks = read_chunks(CONFIG)
    texts = [c["text"] for c in chunks]
    chunk_doc = np.array([c["doc_id"] for c in chunks])
    items = [it for it in load_eval() if it["type"] != "unanswerable"]
    alias_path = ROOT / "store" / "chunks" / f"{CONFIG}_doc_alias.json"
    alias = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    sib, cor = event_siblings(), correction_aliases()
    accepts = {}
    for it in items:
        gold = set(it["gold_doc_ids"])
        acc = set(gold) | set(it.get("also_accept", []))
        for g in gold:
            acc |= set(alias.get(g, [])) | sib.get(g, set()) | cor.get(g, set())
        accepts[it["id"]] = acc

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    n_tok = sum(len(enc.encode(t)) for t in texts)

    rows = []
    for model, spec in MODELS.items():
        full = np.asarray(embed_texts(texts, model), dtype=np.float32)       # 캐시에서 읽음 (없으면 API 호출)
        qfull = np.asarray(embed_texts([it["q"] for it in items], model), dtype=np.float32)
        for d in DIMS[model]:
            mat, q = (full, qfull) if d == spec["dim"] else (_norm(full[:, :d]), _norm(qfull[:, :d]))
            r = rank_all(mat, q, items, chunk_doc, accepts)
            short = model.replace("text-embedding-", "")
            rows.append({"label": f"{short}\n{d}차원", "model": model, "dims": d, **r["overall"],
                         "by_type_hit3": {t: v["hit@3"] for t, v in r["by_type"].items()},
                         "storage_mb": round(len(chunks) * d * 4 / 1e6), "embed_cost_usd": round(n_tok / 1e6 * spec["usd_per_mtok"], 2),
                         "ranks": r["ranks"]})
            o = r["overall"]
            print(f"{model:24s} {d:5d} | Hit@1 {o['hit@1']:.3f} Hit@3 {o['hit@3']:.3f} Hit@5 {o['hit@5']:.3f} "
                  f"Hit@10 {o['hit@10']:.3f} MRR {o['mrr']:.3f} | 저장 {rows[-1]['storage_mb']}MB · 임베딩 ${rows[-1]['embed_cost_usd']}")
        del full
    base = next(r for r in rows if r["model"] == "text-embedding-3-small" and r["dims"] == 1536)
    for r in rows:
        if r is base:
            continue
        gain = [i for i in base["ranks"] if (r["ranks"][i] or 99) <= 3 < (base["ranks"][i] or 99)]
        loss = [i for i in base["ranks"] if (base["ranks"][i] or 99) <= 3 < (r["ranks"][i] or 99)]
        r["vs_small1536"] = {"gain": gain, "loss": loss}
        print(f"  3-small 1536 → {r['label'].replace(chr(10), ' ')}: Hit@3 얻음 {gain} · 잃음 {loss}")
    OUT.write_text(json.dumps({"config": CONFIG, "chunks": len(chunks), "tokens": n_tok, "items": len(items),
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"청크 {len(chunks):,} · 토큰 {n_tok:,} → {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
