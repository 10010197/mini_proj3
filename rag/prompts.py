"""④ 챗봇 계층 — 프롬프트. 답변 원칙은 docs/data_and_use_cases.md §3.3·3.4."""
from __future__ import annotations

REFUSAL_MARK = "[답변 불가]"

SOURCE_LABEL = {"news": "뉴스", "disclosure": "공시", "patent": "특허", "report": "보고서", "policy": "정부 보도자료"}
ROLE_LABEL = {"current": "현재", "leading": "선행", "outlook": "전망"}

SYSTEM = f"""너는 바이오·헬스케어 산업·기술 동향을 분석하는 리서치 어시스턴트다.
아래 [근거]에 있는 문서만 사용해 한국어로 답한다. 지식베이스 기간은 2021-10 ~ 2026-11이다.

규칙
1. 근거에 없는 사실·수치·날짜를 만들지 않는다. 배경지식으로 빈칸을 채우지 않는다.
2. 문장마다 근거 번호를 [1], [2]처럼 붙인다. 번호는 [근거]의 번호만 쓴다.
3. 수치·날짜는 근거에 적힌 그대로 옮기고, 그 근거의 발행일을 함께 적는다.
4. 근거의 종류(공시·특허·보고서·정부 보도자료·뉴스)와 시간축(현재·선행·전망)이 다르면 나눠서 제시한다.
   같은 지표를 출처마다 다르게 제시하면 어느 한쪽을 고르지 말고 **차이와 각 근거의 기준·시점·출처**를 함께 보여준다.
5. 질문 시점(연도·월)이 있으면 그 시점의 근거를 우선하고, 다른 시점 근거를 쓸 때는 그렇다고 밝힌다.
6. 다음 경우에는 답변을 "{REFUSAL_MARK}"로 시작하고, 이유와 (있다면) 가장 가까운 근거를 짧게 덧붙인다.
   - 근거에 질문에 답할 내용이 없거나, 질문한 수치가 비공개·미기재인 경우
   - 지식베이스 기간 밖의 사건이나 전망(예: 2027년 이후 실적)
   - 종목 추천·매수/매도 판단 등 투자 권유, 진단·처방 등 의학적 판단 요청
   일부만 답할 수 있으면 답할 수 있는 부분만 답하고, 없는 부분은 "근거 없음"이라고 밝힌다(이때는 {REFUSAL_MARK}를 붙이지 않는다).
7. 간결하게 쓴다. 핵심 답을 첫 문장에 두고, 필요하면 출처 종류별 글머리표로 정리한다."""

HUMAN = """[근거]
{context}

[질문]
{question}"""

# 대화 맥락이 있는 후속 질문("그 제한은 언제부터?")을 검색용 독립 질문으로 바꾼다. 검색에만 쓰고, 답변은 원래 질문으로 생성한다.
CONDENSE_SYSTEM = """이전 대화와 후속 질문이 주어진다. 후속 질문을 이전 대화 없이도 뜻이 통하는 **검색용 질문 한 문장**으로 다시 써라.
- 대명사·생략(그것, 그 제한, 거기, 그 회사)을 이전 대화의 구체적인 대상(회사명·제품명·제도명·연도)으로 바꾼다.
- 후속 질문이 이전 대화와 무관하면 그대로 돌려준다.
- 답하지 말고 질문만 출력한다."""


def format_context(hits) -> str:
    """검색 결과 → 번호 붙은 근거 블록. 번호는 문서 단위(같은 문서의 청크는 같은 번호)."""
    blocks, num = [], {}
    for h in hits:
        n = num.setdefault(h.doc_id, len(num) + 1)
        m = h.meta
        head = (f"[{n}] ({SOURCE_LABEL.get(m.get('source_type'), m.get('source_type'))}·"
                f"{ROLE_LABEL.get(m.get('evidence_role'), m.get('evidence_role'))}) "
                f"발행일 {str(m.get('published_at', ''))[:10]} · {m.get('publisher', '')}")
        blocks.append(f"{head}\n{h.text}")
    return "\n\n".join(blocks)


def numbered_sources(hits) -> list[dict]:
    """근거 번호 → 출처 정보 (답변 아래 출처 목록, 평가의 인용 채점에 사용)."""
    out, seen = [], {}
    for h in hits:
        if h.doc_id in seen:
            continue
        seen[h.doc_id] = len(seen) + 1
        m = h.meta
        out.append({"n": seen[h.doc_id], "doc_id": h.doc_id, "title": m.get("title", ""), "url": m.get("url", ""),
                    "published_at": str(m.get("published_at", ""))[:10], "publisher": m.get("publisher", ""),
                    "source_type": m.get("source_type", ""), "evidence_role": m.get("evidence_role", ""),
                    "score": h.score})
    return out
