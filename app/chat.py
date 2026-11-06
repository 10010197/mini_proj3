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


def _source(s: dict, key: str) -> None:
    """출처 한 건: 번호·제목(링크)·종류·발행처·발행일 + 근거로 쓰인 원문 일부 + 원문 전체 보기."""
    title = _md(s["title"])
    link = f"[{title}]({s['url']})" if s.get("url") else title
    st.markdown(f"**[{s['n']}]** {link}  \n"
                f"<small>{SOURCE_LABEL.get(s['source_type'], s['source_type'])} · "
                f"{ROLE_LABEL.get(s['evidence_role'], s['evidence_role'])} · {s['publisher']} · "
                f"발행일 {s['published_at']} · 유사도 {s['score']:.3f}</small>", unsafe_allow_html=True)
    for j, ex in enumerate(s.get("excerpts") or []):
        label = " · ".join(x for x in ["원문 일부", ex.get("section", ""), f"유사도 {ex['score']:.3f}"] if x)
        with st.expander(label):
            st.code(ex["text"], language=None, wrap_lines=True)
    if st.toggle("원문 전체 보기", key=f"orig_{key}_{s['doc_id']}"):
        doc = retriever.get_document(s["doc_id"])
        if doc:
            st.code(doc.get("body") or "", language=None, wrap_lines=True, height=320)
        else:
            st.warning("원문 DB에서 문서를 찾지 못했습니다.")


def _sources(msg: dict, key: str) -> None:
    srcs = msg.get("sources") or []
    if not srcs:
        return
    cited = set(msg.get("cited") or [])
    used = [s for s in srcs if s["n"] in cited]
    rest = [s for s in srcs if s["n"] not in cited]
    st.markdown(f"**출처** — 답변에 인용된 근거 {len(used)}건")
    st.caption("AI 답변은 그 자체로 근거가 되지 않습니다. 인용 전에 아래 원문을 직접 확인하세요.")
    for s in used:
        with st.container(border=True):
            _source(s, key)
    if rest:
        with st.expander(f"검색됐지만 인용되지 않은 근거 {len(rest)}건"):
            for s in rest:
                with st.container(border=True):
                    _source(s, key)
    if msg.get("meta"):
        st.caption(msg["meta"])


def _show(msg: dict, key: str) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("refused"):
            st.caption("근거 부족·범위 밖으로 답변을 보류했습니다.")
        st.markdown(_md(msg["content"]))
        _sources(msg, key)


def render(filters: dict) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    top = st.columns([6, 1])
    top[0].caption("답변은 지식베이스 근거만 사용하고, 주장마다 근거 번호 [n]과 출처 종류를 붙입니다. "
                   "답변 아래 출처에서 원문을 바로 확인할 수 있습니다. 사이드바 필터가 검색에 적용됩니다.")
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
        for i, msg in enumerate(st.session_state.messages):
            _show(msg, str(i))

    question = st.chat_input("질문을 입력하세요") or st.session_state.pop("pending", None)
    if not question:
        return
    history = [(m["role"], m["content"]) for m in st.session_state.messages[-2 * HISTORY_TURNS:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with box:
        _show(st.session_state.messages[-1], "pending")
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
