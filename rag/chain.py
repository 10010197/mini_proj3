"""④ 챗봇 계층 — RAG 체인 (LangChain). **챗봇(app/)과 평가 하네스(eval/run_eval.py)가 모두 ask()를 호출한다.**

    질문 ─▶ (대화 맥락이 있으면) 검색용 독립 질문으로 재작성 ─▶ retriever.search (Chroma 조회, 문서 단위 정리)
         ─▶ 근거 블록 ─▶ ChatPromptTemplate(+이전 대화) ─▶ ChatOpenAI ─▶ 답변

반환: {"answer", "refused", "sources"(근거 번호별 출처), "hits"(청크), "usage", "latency_s", "model"}

설정(.env, 선택): CHAT_MODEL (기본 gpt-5.4-mini)
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag import retriever
from rag.prompts import CONDENSE_SYSTEM, HUMAN, REFUSAL_MARK, SYSTEM, format_context, numbered_sources

CHAT_MODEL = retriever.ENV.get("CHAT_MODEL") or "gpt-5.4-mini"
CITE = re.compile(r"\[(\d{1,2})\]")


@lru_cache(maxsize=4)
def llm(model: str = CHAT_MODEL):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=retriever.ENV.get("OPENAI_API_KEY"), timeout=120, max_retries=3)


@lru_cache(maxsize=4)
def build_chain(model: str = CHAT_MODEL):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("history", optional=True),
        ("human", HUMAN),
    ])
    generate = (
        RunnableLambda(lambda x: {"context": format_context(x["hits"]), "question": x["question"],
                                  "history": x.get("history") or []})
        | prompt
        | llm(model)
    )
    return (
        RunnablePassthrough.assign(search_query=RunnableLambda(lambda x: condense(x["question"], x.get("history"), model)))
        .assign(hits=RunnableLambda(lambda x: retriever.search(x["search_query"], **x.get("search", {}))))
        .assign(message=generate)
    )


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


def ask(question: str, history: list[tuple[str, str]] | None = None, model: str = CHAT_MODEL,
        search_kwargs: dict | None = None) -> dict:
    t0 = time.perf_counter()
    out = build_chain(model).invoke({"question": question, "history": _history_messages(history),
                                     "search": search_kwargs or {}})
    msg: AIMessage = out["message"]
    answer = msg.text if isinstance(msg.text, str) else str(msg.content)
    sources = numbered_sources(out["hits"])
    cited = sorted({int(n) for n in CITE.findall(answer)} & {s["n"] for s in sources})
    return {
        "question": question,
        "search_query": out["search_query"],
        "answer": answer.strip(),
        "refused": answer.strip().startswith(REFUSAL_MARK),
        "sources": sources,
        "cited": cited,
        "hits": [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": h.score} for h in out["hits"]],
        "usage": dict(msg.usage_metadata or {}),
        "latency_s": round(time.perf_counter() - t0, 2),
        "model": model,
    }
