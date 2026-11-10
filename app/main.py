"""④ 챗봇 UI (Streamlit) — 검색 탭 + 대화 탭 + 성능 분석 탭(app/dashboard.py).

    streamlit run app/main.py

- 조회 전용: rag.retriever.search(검색)와 rag.chain.ask(대화)만 호출한다.
  문서 임베딩 생성·크롤링·인덱스 쓰기는 하지 않는다. 인덱서 없이 store/chroma/와 store/docs.db만 있으면 구동된다.
- 사이드바 필터(출처 종류·기간·하위 주제·근거 문서 수)는 두 탭에 똑같이 적용된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import chat, dashboard, search_ui  # noqa: E402
from rag import chain, retriever  # noqa: E402
from rag.prompts import SOURCE_LABEL  # noqa: E402

TOPICS = ["ADC", "CDMO", "AI 신약개발", "바이오 투자", "신약 기술수출", "의료 AI 규제", "비대면 진료", "바이오시밀러"]
YEARS = (2021, 2026)


def sidebar_filters() -> dict:
    with st.sidebar:
        st.header("검색 조건")
        types = st.multiselect("출처 종류", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get,
                               placeholder="전체", help="비워 두면 모든 출처")
        y0, y1 = st.slider("발행 연도", *YEARS, value=YEARS)
        topic = st.selectbox("하위 주제", ["전체"] + TOPICS)
        k_docs = st.slider("근거 문서 수", 3, 12, retriever.K_DOCS)
        per_doc = st.slider("문서당 청크 수", 1, 3, retriever.MAX_CHUNKS_PER_DOC)
        st.divider()
        st.caption(f"컬렉션 `{retriever.COLLECTION}`  \n검색 방식 `{retriever.RETRIEVAL_MODE}`  \n"
                   f"생성 모델 `{chain.CHAT_MODEL}`  \n"
                   "지식베이스 기간 2021-10 ~ 2026-11")
    where = retriever.build_where(types or None, y0 if y0 > YEARS[0] else None, y1 if y1 < YEARS[1] else None)
    return {"where": where, "topic": None if topic == "전체" else topic,
            "k_docs": k_docs, "max_chunks_per_doc": per_doc}


def main() -> None:
    st.set_page_config(page_title="바이오·헬스케어 동향 RAG", layout="wide")
    st.title("바이오·헬스케어 산업·기술 동향")
    st.caption("뉴스(현재) · 공시(1차 근거) · 특허(선행 지표) · 보고서(전망) · 정부 보도자료를 함께 검색합니다.")
    try:
        retriever.vectorstore()
        db_error = None
    except Exception as e:
        db_error = e
    filters = sidebar_filters()
    tab_search, tab_chat, tab_dash = st.tabs(["검색", "대화", "성능 분석"])
    if db_error is None:
        with tab_search:
            search_ui.render(filters)
        with tab_chat:
            chat.render(filters)
    else:
        for t in (tab_search, tab_chat):
            t.error(f"벡터 DB를 열 수 없습니다: {db_error}")
    with tab_dash:          # 평가 결과·docs.db만 읽으므로 벡터 DB 없이도 동작
        dashboard.render()


main()
