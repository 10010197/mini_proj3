"""④ 챗봇 UI — 대화 탭: rag.chain.ask(평가 하네스와 같은 체인)로 답변 + 근거 목록."""
from __future__ import annotations

import streamlit as st

from rag import chain, retriever
from rag.prompts import ROLE_LABEL, SOURCE_LABEL

HISTORY_TURNS = 3          # 체인에 넘기는 직전 대화 (질문·답변 쌍)
SUGGESTIONS = [
    "최근 2년간 ADC 기술수출 뉴스가 늘었는데, 국내 ADC 특허 출원도 같이 늘었어?",
    "글로벌 CDMO 시장 규모 전망이 기사와 보고서에서 어떻게 달라?",
    "비대면진료 시 처방이 제한되는 의약품은?",
]


def _md(text: str) -> str:
    return text.replace("$", "\\$")


def _sources(msg: dict) -> None:
    srcs = msg.get("sources") or []
    if not srcs:
        return
    cited = set(msg.get("cited") or [])
    with st.expander(f"근거 {len(srcs)}건 (인용 {len(cited)}건)"):
        for s in srcs:
            mark = "●" if s["n"] in cited else "○"
            title = _md(s["title"])
            link = f"[{title}]({s['url']})" if s.get("url") else title
            st.markdown(f"{mark} **[{s['n']}]** {link}  \n"
                        f"<small>{SOURCE_LABEL.get(s['source_type'], s['source_type'])} · "
                        f"{ROLE_LABEL.get(s['evidence_role'], s['evidence_role'])} · {s['published_at']} · "
                        f"{s['publisher']} · 유사도 {s['score']:.3f}</small>", unsafe_allow_html=True)
        if msg.get("meta"):
            st.caption(msg["meta"])


def _show(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("refused"):
            st.caption("근거 부족·범위 밖으로 답변을 보류했습니다.")
        st.markdown(_md(msg["content"]))
        _sources(msg)


def render(filters: dict) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    top = st.columns([6, 1])
    top[0].caption("답변은 지식베이스 근거만 사용하고, 문장마다 근거 번호 [n]을 붙입니다. 사이드바 필터가 검색에 적용됩니다.")
    if top[1].button("대화 초기화", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    box = st.container(height=560)
    with box:
        if not st.session_state.messages:
            st.markdown("**질문 예시**")
            for q in SUGGESTIONS:
                if st.button(q, key=f"sugg_{q}"):
                    st.session_state.pending = q
                    st.rerun()
        for msg in st.session_state.messages:
            _show(msg)

    question = st.chat_input("질문을 입력하세요") or st.session_state.pop("pending", None)
    if not question:
        return
    history = [(m["role"], m["content"]) for m in st.session_state.messages[-2 * HISTORY_TURNS:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with box:
        _show(st.session_state.messages[-1])
        with st.chat_message("assistant"):
            with st.spinner("근거를 찾고 답변을 쓰는 중…"):
                try:
                    res = chain.ask(question, history=history, search_kwargs={
                        "k_docs": filters["k_docs"], "max_chunks_per_doc": filters["max_chunks_per_doc"],
                        "where": filters["where"], "topic": filters["topic"]})
                except Exception as e:
                    st.error(f"답변 생성 실패: {type(e).__name__}: {e}")
                    st.session_state.messages.pop()
                    return
    tokens = res["usage"].get("total_tokens", 0)
    st.session_state.messages.append({
        "role": "assistant", "content": res["answer"], "refused": res["refused"],
        "sources": res["sources"], "cited": res["cited"],
        "meta": (f"검색 질의: {res['search_query']}  \n" if res["search_query"] != question else "")
                + f"{res['model']} · {retriever.COLLECTION} · {res['latency_s']}초 · 토큰 {tokens:,}",
    })
    st.rerun()
