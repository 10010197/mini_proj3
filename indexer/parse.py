"""② 인덱싱 계층 — 파싱·정제: 문서 종류별로 본문을 정제하고 섹션(제목, 텍스트) 목록으로 나눈다.

원문 추출(HTML·XML·HWPX·PDF → 텍스트)은 수집 계층에서 끝났고, 여기서는 docs.db의 본문을
문서 종류별 구조에 맞게 '섹션'으로 쪼개고 보일러플레이트를 지운다. 청킹(chunk.py)은 섹션 단위로 동작한다.

| source_type | 구조 (EDA에서 확인)                                    | 섹션 분할                         | 제거하는 보일러플레이트 |
|-------------|--------------------------------------------------------|-----------------------------------|-------------------------|
| news        | 줄바꿈 없는 한 덩어리, 리드 문장에 핵심                | 섹션 1개 (본문)                   | 기자명·구독 안내·관련기사 |
| disclosure  | `항목 | 값` 표 줄 (정정 공시는 정정표 + 정정 후 전문) | 수시공시 1개, 사업보고서는 `II.`·`1.`·`가.`·`(1)` 헤딩 계층 | 투자 유의 문구 |
| patent      | 서지 줄(명칭·출원인·IPC) + `요약:` (+ `청구항`)        | 서지+요약 / 청구항                | — |
| report      | `□ ○ ■ 【】 1.` 헤딩, 요약문 + 본문                    | 헤딩 기준                         | `(계속)` 꼬리, 머리 서지 줄 |
| policy      | 제목·부제(`- … -`) + `□`/`○` 문단, 하단 붙임 목록       | `□` 헤딩 기준                     | 붙임·별첨 목록, 의견 제출 안내 |
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Section:
    name: str           # 섹션 경로 (예: "요약", "II. 사업의 내용 > 가. 사업개요", "본문")
    text: str
    kind: str = "text"  # text | table (표 줄 위주 섹션)


@dataclass
class ParsedDoc:
    doc_id: str
    source_type: str
    title: str
    sections: list[Section] = field(default_factory=list)
    removed_chars: int = 0     # 보일러플레이트로 지운 글자 수 (정제 통계용)


# --------------------------------------------------------------------------- #
# 공통 정제
# --------------------------------------------------------------------------- #
WS = re.compile(r"[ \t 　]+")
TABLE_LINE = re.compile(r"\|")


def norm_ws(text: str) -> str:
    lines = [WS.sub(" ", x).strip() for x in (text or "").replace("\r", "").split("\n")]
    return "\n".join(x for x in lines if x)


def split_sentences(text: str) -> list[str]:
    """한국어 문장 분리: 종결 어미+마침표, 물음표·느낌표, 닫는 따옴표 뒤 공백에서 자른다 (숫자 소수점은 유지)."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[다요죠음함됨임까][.?!])\s+|(?<=[.?!][\"'”’])\s+|(?<=[?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# 종류별 파서
# --------------------------------------------------------------------------- #
NEWS_TAIL = [
    re.compile(r"\s*(관련\s?기사|많이 본 뉴스|구독(하기)?|네이버에서 .{0,20}구독|좋아요|공감).*$"),
    re.compile(r"\s*[가-힣]{2,4}\s?(기자|특파원|객원기자)\s*$"),
]
NEWS_INLINE = [
    re.compile(r"\[[^\]]{0,20}(기자|특파원)\]"),
    re.compile(r"\((서울|세종|뉴욕|워싱턴|부산|대전|광주|인천)=[^)]{0,20}\)\s*[가-힣]{2,4}\s?기자\s?="),
]


def parse_news(doc_id: str, title: str, body: str) -> ParsedDoc:
    n0 = len(body)
    t = re.sub(r"\s+", " ", body).strip()
    for p in NEWS_INLINE:
        t = p.sub(" ", t)
    # 꼬리 정리는 마지막 150자 안에서만 (본문 중간의 '공감'·'구독' 같은 단어에서 잘리지 않도록)
    head, tail = t[:-150], t[-150:]
    for p in NEWS_TAIL:
        tail = p.sub("", tail)
    t = head + tail
    t = re.sub(r"\s+", " ", t).strip()
    return ParsedDoc(doc_id, "news", title, [Section("본문", t)], n0 - len(t))


DART_NOTICE = re.compile(
    r"^투자 ?유의사항 \|.*$"                                   # 표 첫 줄로 들어간 경우
    r"|※\s?투자\s?유의\s?사항.*?신중(히|하게)\s?투자하시기\s?바랍니다\.?",  # 본문 문장으로 들어간 경우
    re.M | re.S,
)
DART_HEADING = re.compile(r"^(?:#+\s*)?((?:[IVX]+\.|\d{1,2}\.|[가-하]\.|\(\d{1,2}\))\s*[^|]{2,60})$")


def _dart_level(head: str) -> int:
    if re.match(r"[IVX]+\.", head):
        return 0
    if re.match(r"\d", head):
        return 1
    if re.match(r"[가-하]\.", head):
        return 2
    return 3


def parse_disclosure(doc_id: str, title: str, body: str) -> ParsedDoc:
    """수시공시는 섹션 1개(표 전체). 사업보고서 등 장문은 `II.`/`1.`/`가.`/`(1)` 헤딩 계층으로 나눈다."""
    n0 = len(body)
    t = norm_ws(DART_NOTICE.sub("", body))
    sections: list[Section] = []
    if len(t) <= 6000:
        sections.append(Section("본문", t, _kind(t.split("\n"))))
    else:
        path: list[tuple[int, str]] = []
        cur_name, cur = "본문", []
        for line in t.split("\n"):
            h = DART_HEADING.match(line)
            if not h:
                cur.append(line)
                continue
            if cur:
                sections.append(Section(cur_name, "\n".join(cur), _kind(cur)))
                cur = []
            head = h.group(1).strip()
            lv = _dart_level(head)
            path = [(l, x) for l, x in path if l < lv] + [(lv, head)]
            cur_name = " > ".join(x for _, x in path)
        if cur:
            sections.append(Section(cur_name, "\n".join(cur), _kind(cur)))
        sections = _merge_small(sections, 150)
    out = ParsedDoc(doc_id, "disclosure", title, sections)
    out.removed_chars = n0 - sum(len(s.text) for s in out.sections)
    return out


def _kind(lines: list[str]) -> str:
    return "table" if sum(1 for x in lines if TABLE_LINE.search(x)) >= max(2, len(lines) * 0.5) else "text"


def parse_patent(doc_id: str, title: str, body: str) -> ParsedDoc:
    t = norm_ws(body)
    m = re.search(r"^요약:\s*", t, re.M)
    claims = re.search(r"^(청구항|대표\s?청구항|Claims?)\s*[:：]?\s*", t, re.M)
    biblio = t[:m.start()].strip() if m else t
    abstract = t[m.end():claims.start() if claims and claims.start() > m.end() else None].strip() if m else ""
    sections = [Section("서지·요약", f"{biblio}\n요약: {abstract}" if abstract else biblio)]
    if claims:
        sections.append(Section("청구항", t[claims.end():].strip()))
    return ParsedDoc(doc_id, "patent", title, sections, 0)


REPORT_HEAD_LINES = re.compile(r"^(\[[^\]]+\] .+|발행: .*|키워드: .*)$")
REPORT_TAIL = re.compile(r"\.{5,}\s*\(계속\)\s*$")
REPORT_HEADING = re.compile(r"^(【[^】]+】|□\s?.{2,80}|■\s?.{2,80}|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s?.{2,60}|\d{1,2}\.\s?[^\d].{1,60}|◈.{1,40})$")


def parse_report(doc_id: str, title: str, body: str) -> ParsedDoc:
    n0 = len(body)
    t = REPORT_TAIL.sub("", norm_ws(body))
    # PDF 변환 흔적 ①: 단어마다 줄이 바뀐 문서(줄 길이 중앙값 < 20자) → 줄을 이어 붙이고 □○■·숫자 헤딩 앞에서만 줄바꿈
    lens = sorted(len(x) for x in t.split("\n"))
    if lens and lens[len(lens) // 2] < 20:
        t = re.sub(r"\s*\n\s*", " ", t)
        t = re.sub(r"\s(?=[□○■◈【]|<표|\[ ?그림)", "\n", t)
    # PDF 변환 흔적 ②: 문장부호 앞 공백("있다 ."), 숫자와 단위 사이 공백("1 억 6 천만")
    t = re.sub(r"\s+([.,)%])(?=\s|$)", r"\1", t)
    t = re.sub(r"(\d) (?=(억|천|만|조|%|년|개|건|명|배|달러|원)\b)", r"\1", t)
    t = re.sub(r"([억천만조]) (?=\d)", r"\1", t)
    lines = [x for i, x in enumerate(t.split("\n")) if not (i < 4 and REPORT_HEAD_LINES.match(x))]
    sections, cur_name, cur = [], "본문", []
    for line in lines:
        if REPORT_HEADING.match(line) and len(line) <= 90:
            if cur:
                sections.append(Section(cur_name, "\n".join(cur), _kind(cur)))
            cur_name, cur = line.strip("◈ "), [line]
            continue
        cur.append(line)
    if cur:
        sections.append(Section(cur_name, "\n".join(cur), _kind(cur)))
    out = ParsedDoc(doc_id, "report", title, _merge_small(sections, 150))
    out.removed_chars = n0 - sum(len(s.text) for s in out.sections)
    return out


POLICY_TAIL = re.compile(r"\n(<\s?(별첨|붙임|참고)\s?>|붙임\s?\d|별첨\s?\d|< 의견 제출방법 >|※ ?국민참여입법센터).*$", re.S)


def parse_policy(doc_id: str, title: str, body: str) -> ParsedDoc:
    n0 = len(body)
    t = POLICY_TAIL.sub("", norm_ws(body))
    lines = t.split("\n")
    # 제목·부제 반복(첫 줄들이 제목과 같거나 '- … -' 부제)은 '요지' 섹션으로
    sections, cur_name, cur = [], "요지", []
    for line in lines:
        if line.startswith("□") and cur:
            sections.append(Section(cur_name, "\n".join(cur)))
            cur_name, cur = line[:40].strip("□ "), [line]
            continue
        cur.append(line)
    if cur:
        sections.append(Section(cur_name, "\n".join(cur)))
    out = ParsedDoc(doc_id, "policy", title, _merge_small(sections, 150))
    out.removed_chars = n0 - sum(len(s.text) for s in out.sections)
    return out


def _merge_small(sections: list[Section], min_chars: int) -> list[Section]:
    """너무 짧은 섹션(헤딩만 있는 경우 등)은 다음 섹션에 붙인다."""
    out: list[Section] = []
    carry: Section | None = None
    for s in sections:
        if carry:
            s = Section(carry.name, carry.text + "\n" + s.text, s.kind)
            carry = None
        if len(s.text) < min_chars:
            carry = s
            continue
        out.append(s)
    if carry:
        if out:
            out[-1] = Section(out[-1].name, out[-1].text + "\n" + carry.text, out[-1].kind)
        else:
            out.append(carry)
    return out


PARSERS = {
    "news": parse_news,
    "disclosure": parse_disclosure,
    "patent": parse_patent,
    "report": parse_report,
    "policy": parse_policy,
}


def parse(doc_id: str, source_type: str, title: str, body: str) -> ParsedDoc:
    return PARSERS.get(source_type, parse_news)(doc_id, title, body)
