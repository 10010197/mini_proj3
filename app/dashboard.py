"""④ 성능 분석 탭 — 평가 결과 시각화.

- 조회 전용: eval/results/answer_eval_*.json(하네스 결과)과 store/docs.db(지식베이스 구성)만 읽는다.
  검색·생성·임베딩 호출이 없으므로 벡터 DB가 없어도 그려진다.
- 사이드바 검색 조건은 이 탭에 적용되지 않는다(평가는 필터 없이 79문항 전체로 실행됨).

차트 | 표현 대상
1. 실험 설정별 Hit@1·Hit@3·MRR (기준선 대비)
2. 질문 유형별 정확도 — 취약 유형 식별 + 실패 사례 원문
3. 유사도 점수 분포 — 정답/오답·답변 불가 문항의 최고 코사인 유사도, 임계값 시뮬레이션
4. 문서 종류·시기 분포 — 지식베이스 구성과 편향
5. 응답시간 분포 — 재정렬 적용 전후
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from rag.prompts import SOURCE_LABEL

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
DOCS_DB = ROOT / "store" / "docs.db"

# 실험 설정: (표시 이름, 결과 파일 태그, 설명). 색은 실험(개체)에 고정한다 — 선택을 바꿔도 색이 바뀌지 않게.
BASELINE = "① 기준선 dense"
EXPERIMENTS = [
    (BASELINE, "structure__gpt-5.4-mini_run1", "structure 청킹 · 벡터 검색 · 1차 측정 (11-05)"),
    ("fixed 청킹", "fixed__gpt-5.4-mini", "고정 길이 청킹 · 벡터 검색"),
    ("머리말 제거", "structure_notitle__gpt-5.4-mini", "structure 청킹에서 제목·발행처·날짜 머리말 제거"),
    ("hybrid", "structure__hybrid__gpt-5.4-mini", "벡터 + BM25 (RRF)"),
    ("hybrid_time (이전 프롬프트)", "structure__hybrid_time__gpt-5.4-mini", "hybrid + 질문 연도 목록"),
    ("hybrid_time ★기본", "structure__hybrid_time__gpt-5.4-mini_prompt2", "현재 기본 · 필수 요건 프롬프트 (11-06)"),
    ("hybrid_rerank 1회", "structure__hybrid_rerank__gpt-5.4-mini", "hybrid_time 후보 15문서 → LLM 재정렬"),
    ("hybrid_rerank 2회", "structure__hybrid_rerank__gpt-5.4-mini_run2", "같은 설정 반복 (재정렬은 비결정적)"),
]
EXP_NAMES = [e[0] for e in EXPERIMENTS]
# 기준선은 회색, 나머지는 분류 팔레트 순서대로 (dataviz 기본 팔레트, 인접 쌍 CVD 검증된 순서)
EXP_COLORS = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
EXP_SCALE = alt.Scale(domain=EXP_NAMES, range=EXP_COLORS)

TYPE_LABEL = {"fact": "사실", "comparison": "비교", "temporal": "시점", "synthesis": "종합", "unanswerable": "답변 불가"}
TYPE_ORDER = ["사실", "비교", "시점", "종합"]
SRC_ORDER = ["news", "disclosure", "patent", "report", "policy"]
SRC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRADE_ORDER = ["correct", "partial", "wrong", "과잉 거절"]
GRADE_COLORS = ["#1c5cab", "#86b6ef", "#eb6834", "#8a8984"]   # 정답 진한 파랑 → 부분 연한 파랑, 오답은 대비색


# ---------------------------------------------------------------- 데이터

@st.cache_data(show_spinner=False)
def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """반환: (설정별 요약, 설정×유형 요약, 문항 단위). 파일이 없는 설정은 건너뛴다."""
    summ, by_type, items = [], [], []
    for name, tag, desc in EXPERIMENTS:
        f = RESULTS / f"answer_eval_{tag}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        s, a = d["summary"], d["summary"]["answerable"]
        summ.append({"실험": name, "설명": desc, "검색": d["run"].get("retrieval_mode") or "dense",
                     "컬렉션": d["run"]["collection"].split("__")[0], "파일": f.name,
                     "Hit@1": a["hit@1"], "Hit@3": a["hit@3"], "Hit@6": a["hit@6"], "MRR": a["mrr"],
                     "정확도": a["accuracy"], "근거 충실도": a["grounded"], "인용 정답률": a["cited_gold"],
                     "과잉 거절": a["over_refusal"], "거절 정확도": s["unanswerable"]["refusal_accuracy"],
                     "지연 중앙값(초)": s.get("latency_s_median"),
                     "입력 토큰/문항": round(s["tokens_total"]["input"] / s["n"])})
        for t, v in s["by_type"].items():
            by_type.append({"실험": name, "유형": TYPE_LABEL[t], "문항": v["n"],
                            "정확도": v.get("accuracy", v.get("refusal_accuracy")),
                            "Hit@3": v.get("hit@3"), "MRR": v.get("mrr"), "근거 충실도": v.get("grounded")})
        for it in d["items"]:
            scores = [x["score"] for x in it.get("sources") or [] if x.get("score") is not None]
            j = it.get("judge") or {}
            ans = it["type"] != "unanswerable"
            grade = ("과잉 거절" if it["refused"] else j.get("correctness")) if ans else None
            items.append({"실험": name, "id": it["id"], "유형": TYPE_LABEL[it["type"]], "질문": it["q"],
                          "답변 가능": ans, "정답 순위": it.get("rank"), "거절": it["refused"],
                          "채점": grade, "정확도": it.get("correct"), "근거 충실도": it.get("grounded"),
                          "최고 유사도": max(scores) if scores else None,
                          "1위 유사도": scores[0] if scores else None,
                          "지연(초)": it.get("latency_s"), "채점 이유": j.get("reason"),
                          "정답 기준": it.get("answer_key"), "답변": it.get("answer")})
    return pd.DataFrame(summ), pd.DataFrame(by_type), pd.DataFrame(items)


@st.cache_data(show_spinner=False)
def load_kb() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DOCS_DB.as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("SELECT source_type, substr(published_at, 1, 7) AS ym, in_kb, topics, origin "
                               "FROM documents", con)
    finally:
        con.close()
    df["연도"] = df["ym"].str[:4]
    m = df["ym"].str[5:7].fillna("01").astype(int)
    df["분기"] = df["연도"] + "-Q" + ((m - 1) // 3 + 1).astype(str)
    df["출처"] = df["source_type"].map(SOURCE_LABEL).fillna(df["source_type"])
    return df


def _pct(v: float | None) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.3f}"


# ---------------------------------------------------------------- 1. 설정별 비교

def chart_settings(summ: pd.DataFrame) -> None:
    st.subheader("1. 실험 설정별 검색 성능")
    st.caption("답변 가능 68문항. 체인이 생성 모델에 넘긴 문서 6개 안에서 정답 문서의 순위로 계산한다. "
               "Hit@k·MRR은 같은 설정이면 결정적(재정렬 제외)이고, 정확도는 실행마다 ±0.05까지 흔들린다.")
    c1, c2 = st.columns([3, 2])
    picked = c1.multiselect("비교할 설정", EXP_NAMES, default=[n for n in EXP_NAMES if n in set(summ["실험"])
                                                             and n != "hybrid_time (이전 프롬프트)"],
                            key="dash_exps")
    metrics = c2.multiselect("지표", ["Hit@1", "Hit@3", "Hit@6", "MRR", "정확도", "근거 충실도"],
                             default=["Hit@1", "Hit@3", "MRR"], key="dash_metrics")
    if not picked or not metrics:
        st.info("설정과 지표를 하나 이상 고르세요.")
        return
    sub = summ[summ["실험"].isin(picked)]
    long = sub.melt(id_vars=["실험", "설명"], value_vars=metrics, var_name="지표", value_name="값")
    base_vals = summ.set_index("실험").loc[BASELINE, metrics] if BASELINE in set(summ["실험"]) else None
    if base_vals is not None:
        long["기준선 대비"] = long.apply(lambda r: f"{(r['값'] - base_vals[r['지표']]) * 100:+.1f}%p", axis=1)
    else:
        long["기준선 대비"] = "—"
    order = [n for n in EXP_NAMES if n in picked]
    bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("실험:N", sort=order, axis=alt.Axis(title=None, labels=False, ticks=False)),
        y=alt.Y("값:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".1f", grid=True, title=None)),
        color=alt.Color("실험:N", scale=EXP_SCALE, legend=alt.Legend(title=None, orient="bottom", columns=4, values=order)),
        tooltip=["실험", "설명", "지표", alt.Tooltip("값:Q", format=".3f"), "기준선 대비"],
    )
    labels = bars.mark_text(dy=-6, fontSize=10, color="#52514e").encode(
        text=alt.Text("값:Q", format=".2f"), color=alt.value("#52514e"))
    chart = (bars + labels).properties(width=alt.Step(26), height=260).facet(
        column=alt.Column("지표:N", sort=metrics, header=alt.Header(title=None, labelFontSize=13)),
    ).resolve_scale(x="independent")
    st.altair_chart(chart, width="content")

    # 기준선 대비 표
    tbl = sub.set_index("실험").loc[order, metrics + ["과잉 거절", "거절 정확도", "지연 중앙값(초)"]]
    if base_vals is not None:
        delta = (tbl[metrics] - base_vals) * 100
        show = tbl.copy().astype(object)
        for m in metrics:
            show[m] = [f"{v:.3f} ({d:+.1f}%p)" if n != BASELINE else f"{v:.3f}"
                       for n, v, d in zip(tbl.index, tbl[m], delta[m])]
        st.dataframe(show, width="stretch")
    best = sub.set_index("실험")["Hit@3"].idxmax()
    if base_vals is not None and "Hit@3" in summ:
        b = summ.set_index("실험")
        cur = "hybrid_time ★기본"
        st.markdown(
            f"- 기본 설정 **{cur}**: Hit@3 {b.loc[BASELINE, 'Hit@3']:.3f} → **{b.loc[cur, 'Hit@3']:.3f}**, "
            f"MRR {b.loc[BASELINE, 'MRR']:.3f} → **{b.loc[cur, 'MRR']:.3f}** (하이브리드 + 질문 연도 목록)\n"
            f"- 선택한 설정 중 Hit@3 최고: **{best}** ({sub.set_index('실험').loc[best, 'Hit@3']:.3f}). "
            "재정렬은 사실·시점 문항 순위를 올리지만 종합 문항 근거 충실도가 떨어지고 지연이 +1.4초라 기본값에서 제외했다 "
            "(docs/evaluation.md §7).\n"
            "- 머리말 제거는 Hit@3 −20.6%p로 가장 큰 하락: 청크에 문서 제목·발행처·날짜를 넣는 것이 검색 품질의 핵심이다.")


# ---------------------------------------------------------------- 2. 유형별 정확도

def chart_types(by_type: pd.DataFrame, items: pd.DataFrame) -> None:
    st.subheader("2. 질문 유형별 정확도 — 취약 유형 식별")
    c1, c2 = st.columns([2, 3])
    metric = c1.radio("지표", ["정확도", "Hit@3", "MRR", "근거 충실도"], horizontal=True, key="dash_type_metric")
    exp = c2.selectbox("상세 볼 설정", [n for n in EXP_NAMES if n in set(items["실험"])],
                       index=EXP_NAMES.index("hybrid_time ★기본"), key="dash_type_exp")
    st.caption("정확도: LLM 채점(correct 1 · partial 0.5 · wrong 0, 거절은 0). 문항 수 사실 36 · 비교 9 · 시점 14 · 종합 9 — "
               "비교·종합은 9문항이라 1문항이 약 0.11에 해당한다.")

    hm = by_type[by_type["유형"].isin(TYPE_ORDER)].dropna(subset=[metric])
    present = [n for n in EXP_NAMES if n in set(hm["실험"])]
    base = alt.Chart(hm).encode(
        x=alt.X("유형:N", sort=TYPE_ORDER, axis=alt.Axis(title=None, labelAngle=0, orient="top")),
        y=alt.Y("실험:N", sort=present, axis=alt.Axis(title=None, labelLimit=220)),
    )
    heat = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(f"{metric}:Q", scale=alt.Scale(domain=[0, 1], range=["#cde2fb", "#104281"]),
                        legend=alt.Legend(title=metric, format=".1f")),
        tooltip=["실험", "유형", "문항", alt.Tooltip(f"{metric}:Q", format=".3f")],
    )
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text(f"{metric}:Q", format=".2f"),
        color=alt.condition(f"datum['{metric}'] > 0.55", alt.value("white"), alt.value("#0b0b0b")),
    )
    left, right = st.columns([3, 2])
    left.altair_chart((heat + text).properties(height=34 * len(present) + 30), width="stretch")

    # 선택 설정의 채점 분포 (유형별 누적)
    it = items[(items["실험"] == exp) & items["답변 가능"]]
    cnt = it.groupby(["유형", "채점"]).size().reset_index(name="문항")
    stack = alt.Chart(cnt).mark_bar(stroke="white", strokeWidth=2).encode(
        y=alt.Y("유형:N", sort=TYPE_ORDER, title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("문항:Q", stack="normalize", axis=alt.Axis(format="%", title=None)),
        color=alt.Color("채점:N", scale=alt.Scale(domain=GRADE_ORDER, range=GRADE_COLORS),
                        legend=alt.Legend(title=None, orient="bottom")),
        order=alt.Order("rank:Q"),
        tooltip=["유형", "채점", "문항"],
    ).transform_calculate(rank=f"indexof({json.dumps(GRADE_ORDER)}, datum['채점'])")
    right.markdown(f"**{exp}** · 채점 분포")
    right.altair_chart(stack.properties(height=alt.Step(40)), width="stretch")

    # 취약 유형 진단 (자동 문장)
    tp = by_type[(by_type["실험"] == exp) & by_type["유형"].isin(TYPE_ORDER)].set_index("유형")
    weak = tp["정확도"].idxmin()
    weak_hit = tp["Hit@3"].idxmin()
    per_type_fail = it.assign(검색실패=it["정답 순위"].isna() | (it["정답 순위"] > 3))
    fail_share = per_type_fail.groupby("유형")["검색실패"].mean()
    b_acc = by_type[(by_type["실험"] == BASELINE)].set_index("유형")["정확도"]
    st.markdown(
        f"- **{exp}** 정확도 최저 유형: **{weak}** ({tp.loc[weak, '정확도']:.3f}), "
        f"Hit@3 최저: **{weak_hit}** ({tp.loc[weak_hit, 'Hit@3']:.3f}). "
        f"유형별 Hit@3 실패 비율 — " + " · ".join(f"{t} {fail_share.get(t, 0):.0%}" for t in TYPE_ORDER) + ".\n"
        f"- 기준선에서는 **비교**({b_acc['비교']:.2f})가 가장 낮았다 — 복수 문서를 종합해야 하는 유형의 전형적 패턴이다. "
        f"하이브리드 검색 후 비교는 {b_acc['비교']:.2f} → {tp.loc['비교', '정확도']:.2f}로 올랐지만, "
        f"**종합**은 {b_acc['종합']:.2f} → {tp.loc['종합', '정확도']:.2f}로 최하위에 남았다. 종합 문항은 정답 문서가 여럿이고 "
        "쟁점이 넓어 단일 질의 검색으로는 근거가 한쪽으로 몰린다. 비교 문항의 남은 손실은 검색보다 "
        "**한쪽만 답하는 생성 단계**의 partial이다(C03·C04·C08).\n"
        "- 대응 방향: 종합 → 쟁점별 하위 질의로 나눠 검색(다중 질의), 비교 → 비교 대상별로 근거를 따로 확보하고 "
        "'일부만 답할 수 있으면 그 부분만 답한다' 규칙 강화.")

    with st.expander("실패 사례 직접 확인 (집계만으로는 원인을 알 수 없다)"):
        f1, f2 = st.columns(2)
        t_sel = f1.multiselect("유형", TYPE_ORDER + ["답변 불가"], default=[weak], key="dash_fail_types")
        g_sel = f2.multiselect("채점", GRADE_ORDER, default=["wrong", "partial", "과잉 거절"], key="dash_fail_grades")
        allit = items[items["실험"] == exp]
        fails = allit[allit["유형"].isin(t_sel) & (allit["채점"].isin(g_sel) |
                                                   ((allit["유형"] == "답변 불가") & (allit["정확도"] < 1)))]
        st.dataframe(fails[["id", "유형", "채점", "정답 순위", "최고 유사도", "질문", "정답 기준", "채점 이유"]],
                     width="stretch", hide_index=True,
                     column_config={"정답 순위": st.column_config.NumberColumn(help="비어 있으면 정답 문서가 상위 6개 밖"),
                                    "최고 유사도": st.column_config.NumberColumn(format="%.3f")})
        pick = st.selectbox("답변 원문 보기", fails["id"].tolist(), key="dash_fail_pick") if len(fails) else None
        if pick:
            row = fails[fails["id"] == pick].iloc[0]
            st.markdown(f"**질문** {row['질문']}\n\n**정답 기준** {row['정답 기준']}")
            st.markdown(row["답변"] or "")


# ---------------------------------------------------------------- 3. 유사도 분포

def chart_scores(items: pd.DataFrame) -> None:
    st.subheader("3. 유사도 점수 분포 — 임계값 설정 근거")
    c1, c2, c3 = st.columns([2, 2, 2])
    exp = c1.selectbox("설정", [n for n in EXP_NAMES if n in set(items["실험"])],
                       index=EXP_NAMES.index("hybrid_time ★기본"), key="dash_score_exp")
    grouping = c2.radio("나누는 기준", ["검색 결과", "답변 채점"], horizontal=True, key="dash_score_group")
    which = c3.radio("점수", ["최고 유사도", "1위 유사도"], horizontal=True, key="dash_score_which",
                     help="최고: 근거 6문서 중 가장 높은 코사인 유사도 · 1위: 최종 1위 문서의 유사도(하이브리드는 RRF 순위라 최고와 다를 수 있음)")
    it = items[items["실험"] == exp].dropna(subset=[which]).copy()
    if grouping == "검색 결과":
        it["그룹"] = it.apply(lambda r: "답변 불가 문항" if not r["답변 가능"]
                            else ("정답 문서 상위 6 안" if pd.notna(r["정답 순위"]) else "정답 문서 못 찾음"), axis=1)
        groups = ["정답 문서 상위 6 안", "정답 문서 못 찾음", "답변 불가 문항"]
    else:
        it["그룹"] = it.apply(lambda r: "답변 불가 문항" if not r["답변 가능"] else r["채점"], axis=1)
        groups = ["correct", "partial", "wrong", "과잉 거절", "답변 불가 문항"]
    groups = [g for g in groups if g in set(it["그룹"])]
    lo, hi = float(it[which].min()), float(it[which].max())
    thr = st.slider("유사도 임계값 (미만이면 검색 실패로 처리)", 0.30, 0.80, 0.45, 0.01, key="dash_thr")

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][:len(groups)]
    hist = alt.Chart().mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, stroke="white", strokeWidth=1).encode(
        x=alt.X(f"{which}:Q", bin=alt.Bin(step=0.02, extent=[0.3, 0.8]), title=f"{which} (코사인)",
                scale=alt.Scale(domain=[0.3, 0.8])),
        y=alt.Y("count():Q", title="문항"),
        color=alt.Color("그룹:N", scale=alt.Scale(domain=groups, range=colors), legend=None),
        tooltip=["그룹", alt.Tooltip("count():Q", title="문항")],
    )
    rule = alt.Chart().mark_rule(strokeDash=[4, 3], color="#0b0b0b", strokeWidth=1.5).encode(
        x=alt.datum(thr))
    chart = alt.layer(hist, rule, data=it[["그룹", which]]).properties(height=90, width=420).facet(
        row=alt.Row("그룹:N", sort=groups, header=alt.Header(title=None, labelAngle=0, labelAlign="left", labelFontSize=12)),
    )
    left, right = st.columns([3, 2])
    left.altair_chart(chart, width="content")

    ans, una = it[it["답변 가능"]], it[~it["답변 가능"]]
    blocked_una = (una[which] < thr).sum()
    blocked_ans = (ans[which] < thr).sum()
    blocked_good = ((ans[which] < thr) & ans["채점"].isin(["correct", "partial"])).sum()
    blocked_miss = ((ans[which] < thr) & ans["정답 순위"].isna()).sum()
    right.metric("답변 불가 문항 차단", f"{blocked_una} / {len(una)}")
    right.metric("답변 가능 문항 잘못 차단", f"{blocked_ans} / {len(ans)}",
                 help="그중 correct·partial이던 문항은 정확도 손실, 정답 문서를 못 찾은 문항은 오히려 안전한 거절")
    right.caption(f"잘못 차단된 {blocked_ans}건 중 correct·partial {blocked_good}건 · 정답 문서 못 찾음 {blocked_miss}건")
    med = it.groupby("그룹")[which].median().reindex(groups)
    right.dataframe(med.rename("중앙값").to_frame().style.format("{:.3f}"), width="stretch")
    st.markdown(
        f"- 점수 범위 {lo:.3f} ~ {hi:.3f}. 그룹 간 중앙값 차이가 작고 분포가 크게 겹친다. "
        "text-embedding-3-small의 코사인 유사도는 같은 도메인(바이오·헬스케어) 안에서 0.4~0.7에 몰려, "
        "단일 임계값으로는 '근거 없음'을 가려내기 어렵다.\n"
        "- 답변 불가 문항 다수는 주제가 지식베이스와 겹쳐(예: 범위 밖 연도, 의학적 조언) 유사도가 높게 나온다. "
        "그래서 임계값은 극단적으로 낮은 경우(관련 문서가 아예 없음)만 거르는 안전장치로 두고, "
        "거절 판단은 프롬프트의 거절 규칙(거절 정확도 0.818, 거절 문구 12/12 일치)에 맡기는 것이 맞다.")


# ---------------------------------------------------------------- 4. 지식베이스 구성

def chart_kb() -> None:
    st.subheader("4. 문서 종류·시기 분포 — 지식베이스 구성과 편향")
    if not DOCS_DB.exists():
        st.info("store/docs.db가 없어 지식베이스 구성을 그릴 수 없습니다.")
        return
    df = load_kb()
    c1, c2, c3 = st.columns(3)
    scope = c1.radio("범위", ["지식베이스(임베딩 대상)", "전체 수집"], horizontal=True, key="dash_kb_scope")
    grain = c2.radio("시간 단위", ["연도", "분기"], horizontal=True, key="dash_kb_grain")
    norm = c3.toggle("비율로 보기", key="dash_kb_norm")
    d = df[df["in_kb"] == 1] if scope.startswith("지식베이스") else df
    d = d.dropna(subset=["연도"])
    src_dom = [SOURCE_LABEL[s] for s in SRC_ORDER]
    cnt = d.groupby([grain, "출처"]).size().reset_index(name="문서")
    bars = alt.Chart(cnt).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X(f"{grain}:O", title=None, axis=alt.Axis(labelAngle=-45 if grain == "분기" else 0)),
        y=alt.Y("문서:Q", stack="normalize" if norm else "zero",
                axis=alt.Axis(format="%" if norm else "~s", title=None)),
        color=alt.Color("출처:N", scale=alt.Scale(domain=src_dom, range=SRC_COLORS),
                        legend=alt.Legend(title=None, orient="bottom")),
        order=alt.Order("ord:Q"),
        tooltip=[grain, "출처", "문서"],
    ).transform_calculate(ord=f"indexof({json.dumps(src_dom, ensure_ascii=False)}, datum['출처'])")
    st.altair_chart(bars.properties(height=280), width="stretch")

    left, right = st.columns(2)
    sel = df.groupby("출처").agg(수집=("in_kb", "size"), 지식베이스=("in_kb", "sum")).reindex(src_dom).reset_index()
    sel["선별 비율"] = sel["지식베이스"] / sel["수집"]
    sel_long = sel.melt(id_vars=["출처", "선별 비율"], value_vars=["수집", "지식베이스"], var_name="단계", value_name="문서")
    left.markdown("**출처별 수집 → 지식베이스 선별**")
    left.altair_chart(alt.Chart(sel_long).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("출처:N", sort=src_dom, title=None),
        yOffset=alt.YOffset("단계:N", sort=["수집", "지식베이스"]),
        x=alt.X("문서:Q", title=None, axis=alt.Axis(format="~s")),
        color=alt.Color("단계:N", scale=alt.Scale(domain=["수집", "지식베이스"], range=["#b7d3f6", "#256abf"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["출처", "단계", "문서", alt.Tooltip("선별 비율:Q", format=".0%")],
    ).properties(height=260), width="stretch")

    kb = df[df["in_kb"] == 1]
    topics = kb["topics"].dropna().str.split("; ").explode().str.strip()
    tp = topics[topics != ""].value_counts().rename_axis("하위 주제").reset_index(name="문서")
    right.markdown("**하위 주제별 문서 수 (지식베이스, 중복 태그 허용)**")
    right.altair_chart(alt.Chart(tp).mark_bar(color="#2a78d6", cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("하위 주제:N", sort="-x", title=None), x=alt.X("문서:Q", title=None),
        tooltip=["하위 주제", "문서"],
    ).properties(height=260), width="stretch")

    n = len(kb)
    by_src = kb["출처"].value_counts()
    yr = kb.dropna(subset=["연도"]).groupby("연도").size()
    primary = by_src.reindex(["공시", "정부 보도자료"]).fillna(0).sum() / n
    st.markdown(
        f"- 지식베이스 {n:,}건 중 뉴스 {by_src.get('뉴스', 0) / n:.0%} · 특허 {by_src.get('특허', 0) / n:.0%} · "
        f"1차 근거(공시+정부 보도자료) {primary:.0%}. 제공 뉴스는 키워드 수집물이라 재배포본이 많아, "
        "완전 중복·재배포본 제거 후 분기 층화로 8,001건만 넣었다(뉴스 31,192 → 8,001).\n"
        f"- 시기: 가장 많은 해 {yr.idxmax()}년 {yr.max():,}건, 가장 적은 해 {yr.idxmin()}년 {yr.min():,}건 "
        "(2021년은 10월부터, 2026년은 11월 초까지만 포함). 연도 편향이 시점 문항 검색 실패의 한 원인이라 "
        "질문 연도 목록(hybrid_time)을 추가했다.\n"
        f"- 하위 주제는 {tp.iloc[0]['하위 주제']}·{tp.iloc[1]['하위 주제']}에 몰리고 "
        f"{tp.iloc[-1]['하위 주제']}({tp.iloc[-1]['문서']:,}건)이 가장 적다. 적은 주제의 질문은 근거 후보가 적어 "
        "검색 실패 시 대체 문서가 없다(docs/kb_composition.md).")


# ---------------------------------------------------------------- 5. 응답시간

def chart_latency(items: pd.DataFrame, summ: pd.DataFrame) -> None:
    st.subheader("5. 응답시간 분포 — 재정렬 적용 전후")
    default = [n for n in ["① 기준선 dense", "hybrid", "hybrid_time ★기본", "hybrid_rerank 1회", "hybrid_rerank 2회"]
               if n in set(items["실험"])]
    picked = st.multiselect("설정", [n for n in EXP_NAMES if n in set(items["실험"])], default=default,
                            key="dash_lat_exps")
    if not picked:
        return
    it = items[items["실험"].isin(picked)].dropna(subset=["지연(초)"])
    order = [n for n in EXP_NAMES if n in picked]
    y = alt.Y("실험:N", sort=order, title=None)
    box = alt.Chart(it).mark_boxplot(size=18, extent=1.5, median={"color": "#0b0b0b"}, opacity=0.35).encode(
        y=y, x=alt.X("지연(초):Q", title="문항당 응답시간(초, 검색+생성)"),
        color=alt.Color("실험:N", scale=EXP_SCALE, legend=None))
    dots = alt.Chart(it).mark_circle(size=40, opacity=0.75, stroke="white", strokeWidth=1).encode(
        y=y, x="지연(초):Q", yOffset=alt.YOffset("jit:Q"),
        color=alt.Color("실험:N", scale=EXP_SCALE, legend=None),
        tooltip=["실험", "id", "유형", alt.Tooltip("지연(초):Q", format=".2f")],
    ).transform_calculate(jit="random() * 16 - 8")
    st.altair_chart((box + dots).properties(height=60 * len(order)), width="stretch")
    s = summ.set_index("실험").loc[order, ["지연 중앙값(초)", "입력 토큰/문항", "Hit@3", "정확도"]]
    p90 = it.groupby("실험")["지연(초)"].quantile(0.9).reindex(order).round(2)
    s.insert(1, "지연 p90(초)", p90)
    st.dataframe(s, width="stretch")
    st.markdown(
        "- 재정렬(hybrid_rerank)은 LLM 호출이 1회 늘어 지연 중앙값이 약 2.1초 → 3.5초(+1.4초)다. "
        "재정렬 호출의 입력(문항당 약 1.1만 토큰)은 위 토큰 집계에 포함되지 않는다.\n"
        "- 하이브리드(BM25·연도 목록 추가)는 지연에 거의 영향이 없다.\n"
        "- 캐싱은 아직 구현하지 않았다(심화 확장). 적용 시 같은 하네스로 이 차트에 '캐시 적용' 설정을 추가해 비교한다.")


# ---------------------------------------------------------------- 진입점

def render() -> None:
    summ, by_type, items = load_runs()
    if summ.empty:
        st.info("eval/results/에 평가 결과가 없습니다. `python -m eval.run_eval`로 먼저 평가를 실행하세요.")
        return
    st.caption(f"평가셋 79문항(답변 가능 68 · 답변 불가 11) · 생성 gpt-5.4-mini · 채점 gpt-5.4 · "
               f"실험 {len(summ)}개 · 근거: docs/evaluation.md. 사이드바 검색 조건은 이 탭에 적용되지 않는다.")
    k = summ.set_index("실험")
    if {BASELINE, "hybrid_time ★기본"} <= set(k.index):
        b, c = k.loc[BASELINE], k.loc["hybrid_time ★기본"]
        cols = st.columns(5)
        for col, m in zip(cols, ["Hit@1", "Hit@3", "MRR", "정확도", "과잉 거절"]):
            col.metric(f"{m} (현재 기본)", f"{c[m]:.3f}", f"{(c[m] - b[m]) * 100:+.1f}%p vs 기준선",
                       delta_color="inverse" if m == "과잉 거절" else "normal")
    sections = ["설정별 비교", "질문 유형별", "유사도 분포", "지식베이스 구성", "응답시간"]
    sec = st.segmented_control("차트", sections, default=sections[0], key="dash_section") or sections[0]
    st.divider()
    if sec == "설정별 비교":
        chart_settings(summ)
    elif sec == "질문 유형별":
        chart_types(by_type, items)
    elif sec == "유사도 분포":
        chart_scores(items)
    elif sec == "지식베이스 구성":
        chart_kb()
    else:
        chart_latency(items, summ)
    with st.expander("실험 설정·결과 파일"):
        st.dataframe(summ[["실험", "설명", "컬렉션", "검색", "파일"]], width="stretch", hide_index=True)
