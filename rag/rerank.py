"""④ 챗봇 계층 — LLM 재정렬 (retriever의 mode="hybrid_rerank"에서 쓴다).

hybrid_time 후보를 문서 RERANK_DOCS개(문서당 최대 2청크)까지 넓게 모은 뒤, LLM이 청크마다 질문과의 관련도를
0~10점으로 매긴다. 문서 점수는 그 문서 청크 점수의 최댓값이고, 동점이면 원래(RRF) 순위를 따른다.
대상: 정답 문서를 찾긴 했지만 4위 이하로 밀린 문항 (docs/evaluation.md §6.4, §7).

- GPU가 없어 로컬 cross-encoder 대신 API(기본 gpt-5.4-mini) 한 번 호출로 모든 후보를 함께 채점한다.
- LLM 호출이 실패하면 원래 순서를 그대로 쓴다 (챗봇이 멈추지 않게).

설정(.env, 선택): RERANK_MODEL (기본 CHAT_MODEL → gpt-5.4-mini)
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel, Field

from rag.prompts import SOURCE_LABEL

log = logging.getLogger(__name__)

PASSAGE_CHARS = 700       # 청크당 LLM에 보여주는 글자 수 (머리말 포함)

SYSTEM = """너는 검색 결과 재정렬기다. 질문과 번호 붙은 문단들이 주어진다.
각 문단이 **질문에 답하는 데 직접 쓰일 근거를 담고 있는지** 0~10점으로 매겨라.
- 10: 질문이 묻는 사실·수치·날짜를 직접 담고 있다 (질문의 회사·제품·제도·시점이 일치)
- 6~8: 질문의 대상과 같지만 일부만 답하거나, 답의 일부 쟁점(비교·종합 질문의 한 축)을 담고 있다
- 3~5: 같은 주제지만 다른 회사·다른 시점·다른 제품이다
- 0~2: 관련 없다
질문에 연도·월이 있으면 그 시점의 문서를 우선한다. 같은 공시의 다른 사업연도·정정 전후를 구분한다.
모든 문단 번호에 대해 점수를 하나씩 매긴다."""


class Score(BaseModel):
    i: int = Field(description="문단 번호")
    score: int = Field(ge=0, le=10)


class Scores(BaseModel):
    scores: list[Score]


@lru_cache(maxsize=2)
def _llm(model: str):
    from langchain_openai import ChatOpenAI

    from rag import retriever

    return ChatOpenAI(model=model, api_key=retriever.ENV.get("OPENAI_API_KEY"), timeout=120, max_retries=3) \
        .with_structured_output(Scores)


def model_name() -> str:
    from rag import retriever

    return retriever.ENV.get("RERANK_MODEL") or retriever.ENV.get("CHAT_MODEL") or "gpt-5.4-mini"


def _passage(n: int, h) -> str:
    m = h.meta
    head = f"[{n}] {SOURCE_LABEL.get(m.get('source_type'), m.get('source_type', ''))} | {m.get('published_at', '')[:10]}"
    return f"{head}\n{' '.join(h.text.split())[:PASSAGE_CHARS]}"


def score(query: str, hits: list) -> list[float] | None:
    """청크별 관련도(0~10). 실패하면 None."""
    if not hits:
        return []
    body = "\n\n".join(_passage(n, h) for n, h in enumerate(hits, 1))
    try:
        out: Scores = _llm(model_name()).invoke([("system", SYSTEM), ("human", f"[질문]\n{query}\n\n[문단]\n{body}")])
    except Exception as e:  # noqa: BLE001 — 재정렬 실패는 원래 순서로 대체
        log.warning("rerank failed: %s", e)
        return None
    got = {s.i: float(s.score) for s in out.scores}
    return [got.get(n, -1.0) for n in range(1, len(hits) + 1)]     # 점수를 빠뜨린 문단은 맨 뒤


def rerank(query: str, hits: list, k_docs: int) -> list:
    """retriever.Hit 목록(문서 순위순) → 재정렬한 상위 k_docs개 문서의 청크. 각 Hit.meta['rerank']에 점수를 남긴다."""
    scores = score(query, hits)
    if scores is None:          # 실패: 원래 순서 그대로, 점수는 None으로 남겨 실제 0점과 구분한다
        keep = list(dict.fromkeys(h.doc_id for h in hits))[:k_docs]
        for h in hits:
            h.meta["rerank"] = None
        return [h for h in hits if h.doc_id in keep]
    order: list[str] = []
    best: dict[str, float] = {}
    for h, s in zip(hits, scores):
        h.meta["rerank"] = s
        if h.doc_id not in best:
            order.append(h.doc_id)
            best[h.doc_id] = s
        else:
            best[h.doc_id] = max(best[h.doc_id], s)
    rank = {d: r for r, d in enumerate(order)}
    keep = sorted(order, key=lambda d: (-best[d], rank[d]))[:k_docs]
    return [h for d in keep for h in hits if h.doc_id == d]
