"""④ 챗봇 계층 — RAG 체인 (LangChain). **챗봇(app/)과 평가 하네스(eval/run_eval.py)가 같은 단계를 쓴다.**

    질문 ─▶ (대화 맥락이 있으면) 검색용 독립 질문으로 재작성 ─▶ retriever.search (Chroma 조회, 문서 단위 정리)
         ─▶ 근거 블록 ─▶ ChatPromptTemplate(+이전 대화) ─▶ ChatOpenAI ─▶ 답변

- ask(): 한 번에 실행 (평가 하네스).
- retrieve() → stream_answer() → finalize(): 같은 단계를 나눠 실행 (챗봇: 검색 진행 표시 + 스트리밍 출력).

반환: {"answer", "refused", "sources"(근거 번호별 출처), "hits"(청크), "context"(LLM이 받은 근거 블록), "usage", "latency_s", "model"}

설정(.env, 선택): CHAT_MODEL (기본 gpt-5.4-mini)
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from functools import lru_cache

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag import retriever
from rag.prompts import CONDENSE_SYSTEM, HUMAN, REFUSAL_MARK, SYSTEM, format_context, numbered_sources

CHAT_MODEL = retriever.ENV.get("CHAT_MODEL") or "gpt-5.4-mini"
CITE = re.compile(r"\[(\d{1,2})\]")


@lru_cache(maxsize=4)
def llm(model: str = CHAT_MODEL):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=retriever.ENV.get("OPENAI_API_KEY"), timeout=120, max_retries=3,
                      stream_usage=True)       # 스트리밍에서도 토큰 사용량을 받는다


@lru_cache(maxsize=4)
def build_chain(model: str = CHAT_MODEL):
    """근거 블록 + 질문 (+이전 대화) → 답변 메시지. invoke()와 stream()을 모두 지원한다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("history", optional=True),
        ("human", HUMAN),
    ])
    return prompt | llm(model)


def condense(question: str, history: list | None, model: str = CHAT_MODEL) -> str:
    """후속 질문 → 검색용 독립 질문. 대화 맥락이 없으면(평가 하네스 포함) LLM을 부르지 않고 그대로 쓴다."""
    if not history:
        return question
    msgs = [("system", CONDENSE_SYSTEM), *history, ("human", f"[후속 질문]\n{question}")]
    out = llm(model).invoke(msgs)
    text = (out.text if isinstance(out.text, str) else str(out.content)).strip()
    return text or question


def _history_messages(history: list[tuple[str, str]] | None) -> list:
    msgs = []
    for role, text in history or []:
        msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
    return msgs


def retrieve(question: str, history: list[tuple[str, str]] | None = None, model: str = CHAT_MODEL,
             search_kwargs: dict | None = None) -> dict:
    """1단계: (후속 질문이면) 검색용 질문 재작성 → 검색. 반환 state는 stream_answer()·finalize()에 넘긴다."""
    t0 = time.perf_counter()
    msgs = _history_messages(history)
    search_query = condense(question, msgs, model)
    hits = retriever.search(search_query, **(search_kwargs or {}))
    return {"question": question, "history": msgs, "search_query": search_query, "hits": hits, "t0": t0}


def _inputs(state: dict) -> dict:
    return {"context": format_context(state["hits"]), "question": state["question"], "history": state["history"]}


def stream_answer(state: dict, model: str = CHAT_MODEL) -> Iterator[str]:
    """2단계(스트리밍): 답변 텍스트 조각을 차례로 내보낸다. 다 끝나면 state["message"]에 전체 메시지가 남는다."""
    full: AIMessageChunk | None = None
    for chunk in build_chain(model).stream(_inputs(state)):
        full = chunk if full is None else full + chunk
        text = chunk.text if isinstance(chunk.text, str) else str(chunk.content)
        if text:
            yield text
    state["message"] = full or AIMessage("")


def generate(state: dict, model: str = CHAT_MODEL) -> dict:
    """2단계(한 번에): 답변 메시지를 state["message"]에 넣는다."""
    state["message"] = build_chain(model).invoke(_inputs(state))
    return state


def finalize(state: dict, model: str = CHAT_MODEL) -> dict:
    """3단계: 답변 메시지 → 결과 dict (출처 번호, 인용, 거절 여부, 사용량)."""
    msg = state["message"]
    answer = msg.text if isinstance(msg.text, str) else str(msg.content)
    sources = numbered_sources(state["hits"])
    cited = sorted({int(n) for n in CITE.findall(answer)} & {s["n"] for s in sources})
    return {
        "question": state["question"],
        "search_query": state["search_query"],
        "answer": answer.strip(),
        "refused": answer.strip().startswith(REFUSAL_MARK),
        "sources": sources,
        "cited": cited,
        "hits": [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": h.score, "rerank": h.meta.get("rerank")}
                 for h in state["hits"]],
        "context": format_context(state["hits"]),      # LLM이 실제로 받은 근거 블록 (평가 하네스의 채점 입력)
        "usage": dict(getattr(msg, "usage_metadata", None) or {}),
        "latency_s": round(time.perf_counter() - state["t0"], 2),
        "model": model,
    }


def ask(question: str, history: list[tuple[str, str]] | None = None, model: str = CHAT_MODEL,
        search_kwargs: dict | None = None) -> dict:
    state = retrieve(question, history, model, search_kwargs)
    return finalize(generate(state, model), model)
