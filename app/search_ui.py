"""④ 챗봇 UI — 검색 탭: 질의 → 상위 문서·청크·유사도·메타데이터, 원문 보기(docs.db 조회)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rag import retriever
from rag.prompts import ROLE_LABEL, SOURCE_LABEL

EXAMPLES = [
    "리가켐바이오의 최근 기술이전 계약 상대방과 계약 규모는?",
    "글로벌 CDMO 시장 규모 전망",
    "비대면 진료 처방 제한 의약품",
    "항체-약물 접합체 특허 출원",
]


def _group(hits: list) -> list[tuple[str, list]]:
    order, by_doc = [], {}
    for h in hits:
        if h.doc_id not in by_doc:
            by_doc[h.doc_id] = []
            order.append(h.doc_id)
        by_doc[h.doc_id].append(h)
    return [(d, by_doc[d]) for d in order]


def _md(text: str) -> str:
    return text.replace("$", "\\$")


def render(filters: dict) -> None:
    c1, c2 = st.columns([5, 1])
    query = c1.text_input("검색어", key="search_q", placeholder="예: " + EXAMPLES[0], label_visibility="collapsed")
    c2.button("검색", type="primary", width="stretch", key="search_btn")
    st.caption("예시: " + " · ".join(EXAMPLES))
    if not query.strip():
        st.info("검색어를 입력하세요. 왼쪽 사이드바에서 출처 종류·기간·주제를 거를 수 있습니다.")
        return

    with st.spinner("검색 중…"):
        hits = retriever.search(query, k_docs=filters["k_docs"], max_chunks_per_doc=filters["max_chunks_per_doc"],
                                where=filters["where"], topic=filters["topic"])
    if not hits:
        st.warning("조건에 맞는 문서가 없습니다. 필터를 넓혀 보세요.")
        return

    groups = _group(hits)
    rows = [{"순위": i, "유사도": hs[0].score, "출처": SOURCE_LABEL.get(hs[0].meta.get("source_type"), ""),
             "시간축": ROLE_LABEL.get(hs[0].meta.get("evidence_role"), ""),
             "발행일": str(hs[0].meta.get("published_at", ""))[:10], "발행처": hs[0].meta.get("publisher", ""),
             "제목": hs[0].meta.get("title", ""), "청크": len(hs)}
            for i, (_, hs) in enumerate(groups, 1)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"유사도": st.column_config.ProgressColumn("유사도", min_value=0.0, max_value=1.0, format="%.3f")})

    for i, (doc_id, hs) in enumerate(groups, 1):
        m = hs[0].meta
        with st.container(border=True):
            title = m.get("title", doc_id)
            st.markdown(f"**{i}. [{_md(title)}]({m.get('url', '')})**" if m.get("url") else f"**{i}. {_md(title)}**")
            st.caption(" · ".join(x for x in [
                SOURCE_LABEL.get(m.get("source_type"), ""), ROLE_LABEL.get(m.get("evidence_role"), ""),
                str(m.get("published_at", ""))[:10], m.get("publisher", ""), m.get("topics", ""),
                f"유사도 {hs[0].score:.3f}", f"`{doc_id}`"] if x))
            for h in hs:
                with st.expander(f"청크 #{h.meta.get('chunk_index', 0) + 1}/{h.meta.get('n_chunks', '?')} · "
                                 f"{h.meta.get('section', '')} · 유사도 {h.score:.3f}", expanded=(h is hs[0])):
                    st.code(h.text, language=None, wrap_lines=True)
            if st.toggle("원문 보기", key=f"orig_{doc_id}"):
                doc = retriever.get_document(doc_id)
                if doc:
                    st.caption(f"수집 일시 {doc.get('loaded_at', '')} · 본문 {len(doc.get('body') or ''):,}자 (store/docs.db)")
                    st.code(doc.get("body") or "", language=None, wrap_lines=True)
                else:
                    st.warning("원문 DB에서 문서를 찾지 못했습니다.")
