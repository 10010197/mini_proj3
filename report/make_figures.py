"""리포트 그림 생성 — 성능 분석 탭과 같은 원자료(eval/results, eval/experiments.json, store/docs.db)에서 PNG를 만든다.

    python report/make_figures.py       → report/figures/*.png
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 150})
PAL = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COLS = lambda n: (PAL + ["#104281", "#b8b7b2", "#b8b7b2"])[:n]  # noqa: E731  9번째 실험(임베딩 3-large)은 진한 파랑
CUR = "structure__hybrid_time__gpt-5.4-mini_prompt2"
TYPES = [("fact", "사실"), ("comparison", "비교"), ("temporal", "시점"), ("synthesis", "종합")]


def load(tag: str) -> dict:
    return json.loads((RES / f"answer_eval_{tag}.json").read_text(encoding="utf-8"))


def experiments() -> list[dict]:
    reg = json.loads((ROOT / "eval" / "experiments.json").read_text(encoding="utf-8"))["experiments"]
    return [e for e in reg if e["kind"] == "answer" and e.get("dashboard", True)]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("→", (OUT / name).relative_to(ROOT).as_posix())


def fig_settings() -> None:
    exps = experiments()
    metrics = ["hit@1", "hit@3", "mrr"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axes, metrics):
        vals = [load(e["tag"])["summary"]["answerable"][m] for e in exps]
        x = np.arange(len(exps))
        bars = ax.bar(x, vals, color=COLS(len(exps)), width=0.75, edgecolor="white", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#52514e")
        ax.set_title({"hit@1": "Hit@1", "hit@3": "Hit@3", "mrr": "MRR"}[m])
        ax.set_xticks([])
        ax.set_ylim(0, 0.95)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLS(len(exps))]
    fig.legend(handles, [e["name"] for e in exps], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("실험 설정별 검색 성능 (답변 가능 68문항)", x=0.01, ha="left", fontsize=12)
    save(fig, "fig1_settings.png")


def fig_types() -> None:
    exps = experiments()
    mat = np.array([[load(e["tag"])["summary"]["by_type"][t]["accuracy"] for t, _ in TYPES] for e in exps])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [l for _, l in TYPES])
    ax.set_yticks(range(len(exps)), [e["name"] for e in exps])
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] > 0.55 else "#0b0b0b", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label="정확도 (LLM 채점)")
    ax.set_title("질문 유형별 정확도 — 실험 설정별", loc="left")
    save(fig, "fig2_type_accuracy.png")


def fig_human() -> None:
    agr = json.loads((RES / f"manual_agreement__{CUR}.json").read_text(encoding="utf-8"))
    bt = agr["by_type"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    h = [bt[t]["manual"] for t, _ in TYPES]
    l = [bt[t]["llm"] for t, _ in TYPES]
    b1 = ax.bar(x - 0.19, h, 0.36, color="#2a78d6", label="사람 채점", edgecolor="white")
    b2 = ax.bar(x + 0.19, l, 0.36, color="#eb6834", label="LLM 채점 (같은 문항)", edgecolor="white")
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(x, [f"{lab} ({bt[t]['n']})" for t, lab in TYPES])
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"사람 vs LLM 채점 — 블라인드 표본 {agr['n_scored']}문항 (3-small 설정)", loc="left")
    save(fig, "fig3_human_vs_llm.png")


def fig_similarity() -> None:
    run = load(CUR)
    groups = {"정답 문서 상위 6 안": [], "정답 문서 못 찾음": [], "답변 불가 문항": []}
    for it in run["items"]:
        sc = [s["score"] for s in it["sources"] if s.get("score") is not None]
        if not sc:
            continue
        k = "답변 불가 문항" if it["type"] == "unanswerable" else ("정답 문서 상위 6 안" if it.get("rank") else "정답 문서 못 찾음")
        groups[k].append(max(sc))
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6), sharex=True)
    bins = np.arange(0.36, 0.80, 0.02)
    for ax, (k, v), c in zip(axes, groups.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.hist(v, bins=bins, color=c, edgecolor="white")
        ax.axvline(np.median(v), color="#0b0b0b", lw=1, ls=":")
        ax.text(0.01, 0.8, f"{k}  n={len(v)} · 중앙값 {np.median(v):.3f}", transform=ax.transAxes, ha="left", fontsize=9)
        ax.set_yticks([0, 5, 10])
    axes[-1].set_xlabel("근거 6문서 중 최고 코사인 유사도")
    axes[0].set_title("유사도 점수 분포 — 세 그룹이 크게 겹친다", loc="left")
    save(fig, "fig4_similarity.png")


def fig_kb() -> None:
    con = sqlite3.connect(f"file:{(ROOT / 'store' / 'docs.db').as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT substr(published_at,1,4), source_type, count(*) FROM documents WHERE in_kb=1 GROUP BY 1,2").fetchall()
    con.close()
    years = sorted({r[0] for r in rows})
    srcs = [("news", "뉴스"), ("disclosure", "공시"), ("patent", "특허"), ("report", "보고서"), ("policy", "정부 보도자료")]
    d = {(y, s): n for y, s, n in rows}
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    bottom = np.zeros(len(years))
    for (s, lab), c in zip(srcs, PAL[1:6]):
        v = np.array([d.get((y, s), 0) for y in years])
        ax.bar(years, v, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=1)
        bottom += v
    for i, t in enumerate(bottom):
        ax.text(i, t + 40, f"{int(t):,}", ha="center", fontsize=8, color="#52514e")
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_title("지식베이스 문서 — 연도·출처별 (15,379건, 2021은 10월부터·2026은 11월 초까지)", loc="left", fontsize=10)
    save(fig, "fig5_kb_composition.png")


def fig_latency() -> None:
    names = [("① 기준선 dense", "structure__gpt-5.4-mini_run1"), ("hybrid", "structure__hybrid__gpt-5.4-mini"),
             ("hybrid_time (3-small)", CUR), ("hybrid_rerank 1회", "structure__hybrid_rerank__gpt-5.4-mini"),
             ("hybrid_rerank 2회", "structure__hybrid_rerank__gpt-5.4-mini_run2"),
             ("3-large ★기본", "structure__3large__hybrid_time__gpt-5.4-mini_prompt2")]
    data = [[it["latency_s"] for it in load(t)["items"] if it.get("latency_s") is not None] for _, t in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, medianprops={"color": "#0b0b0b"})
    for p, c in zip(bp["boxes"], [PAL[0], PAL[3], PAL[5], PAL[6], PAL[7], "#104281"]):
        p.set_facecolor(c)
        p.set_alpha(0.55)
    ax.set_yticks(range(1, len(names) + 1), [n for n, _ in names])
    for i, v in enumerate(data, 1):
        ax.text(max(v) + 0.1, i, f"중앙값 {np.median(v):.1f}초", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("문항당 응답시간 (초, 검색 + 생성)")
    ax.set_title("응답시간 — 재정렬·임베딩 교체 전후", loc="left")
    save(fig, "fig6_latency.png")


def fig_layers() -> None:
    fl = json.loads((RES / f"failure_layers__{CUR}.json").read_text(encoding="utf-8"))
    order = ["데이터", "전처리", "인덱싱", "검색", "생성", "평가셋"]
    n = [len(fl["by_layer"].get(k, [])) for k in order]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bars = ax.barh(order[::-1], n[::-1], color="#2a78d6", edgecolor="white")
    for b, v in zip(bars, n[::-1]):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    ax.set_xlim(0, max(n) * 1.15)
    ax.set_title(f"실패 원인 계층 — 실패 {fl['n_failed']}건 (3-small 설정, 사람 채점 기준 실행)", loc="left")
    save(fig, "fig7_failure_layers.png")


def fig_embedding() -> None:
    p = RES / "embedding_comparison.json"
    if not p.exists():
        print("embedding_comparison.json 없음 — fig8 생략")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["rows"]
    labels = [r["label"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(rows))
    for i, (m, c) in enumerate([("hit@3", "#2a78d6"), ("mrr", "#eb6834"), ("hit@10", "#1baf7a")]):
        bs = ax.bar(x + (i - 1) * 0.26, [r[m] for r in rows], 0.25, color=c, label={"hit@3": "Hit@3", "mrr": "MRR", "hit@10": "Hit@10"}[m],
                    edgecolor="white")
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("임베딩 모델·차원별 검색 성능 (structure 청킹, 벡터 검색 단독, 68문항)", loc="left", fontsize=10)
    save(fig, "fig8_embedding.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (fig_settings, fig_types, fig_human, fig_similarity, fig_kb, fig_latency, fig_layers, fig_embedding):
        f()
