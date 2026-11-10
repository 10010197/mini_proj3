"""② 인덱싱 계층 — 청킹: 문서 종류별 분할 + 청크 메타데이터.

비교용으로 청킹 설정을 여러 개 둔다 (eval/retrieval_eval.py에서 Hit@3으로 비교, 결과는 docs/indexing.md).

  fixed       모든 문서를 같은 방식으로: 섹션 구분 없이 800자 창, 100자 겹침 (문장 경계 우선). 제목 머리말 포함
  structure   문서 종류별 분할 (아래 표). 제목·출처·날짜·섹션 머리말 포함
  structure_notitle  structure와 같은 분할, 머리말 없이 본문만 → '청크에 제목 포함' 효과 측정용

structure 설정 — EDA에서 확인한 문서 구조를 근거로 정함
| 종류       | 단위        | 목표 길이 | 겹침     | 근거 |
|------------|-------------|-----------|----------|------|
| news       | 문장        | 600자     | 1문장    | 중앙값 1,466자 → 2~3청크. 리드에 핵심이 있어 짧게 |
| policy     | 문장(□ 문단)| 500자     | 1문장    | 구조적이고 핵심이 앞부분 → 짧게 + 겹침 |
| report     | 헤딩 섹션   | 900자     | 1문장    | 헤딩(□·■·1.)이 의미 단위. 표는 행 단위로 따로 |
| disclosure | 표 행/문장  | 1,000자   | 없음     | 수시공시는 대부분 1청크. 사업보고서는 헤딩 섹션별, 표 행을 자르지 않음 |
| patent     | 항목        | 서지·요약 1청크, 청구항 1,000자 | 없음 | 항목 구조 고정 |
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import pandas as pd

from indexer.parse import ParsedDoc, Section, parse, split_sentences

SOURCE_LABEL = {"news": "뉴스", "disclosure": "공시", "patent": "특허", "report": "보고서", "policy": "정부 보도자료"}


@dataclass(frozen=True)
class Profile:
    target: int
    overlap_units: int
    unit: str            # sentence | line


STRUCTURE = {
    "news": Profile(600, 1, "sentence"),
    "policy": Profile(500, 1, "sentence"),
    "report": Profile(900, 1, "sentence"),
    "disclosure": Profile(1000, 0, "line"),
    "patent": Profile(1000, 0, "line"),
}
FIXED_SIZE, FIXED_OVERLAP = 800, 100
MIN_CHUNK = 120          # 이보다 짧은 꼬리 청크는 앞 청크에 붙인다
MAX_CHUNK = 2000         # 한 단위가 이보다 길면 강제로 자른다
LONG_LINE = 1200         # 표·목록 한 줄이 이보다 길면 문장으로 나눈다

CONFIGS = ("fixed", "structure", "structure_notitle")

# 청크 메타데이터 (VectorDB에 함께 저장, 모두 스칼라 값)
METADATA_FIELDS = [
    "chunk_id", "doc_id", "chunk_index", "n_chunks",
    "source_type", "evidence_role", "origin", "title", "publisher", "published_at", "year", "quarter", "url",
    "section", "content_kind", "topics", "is_outlook", "event_id", "company", "category",
    "char_len", "config",
]


# --------------------------------------------------------------------------- #
# 분할 도구
# --------------------------------------------------------------------------- #
def _units(sec: Section, unit: str) -> list[str]:
    if sec.kind == "table" or unit == "line":
        out = []
        for line in sec.text.split("\n"):
            out.extend(split_sentences(line) if len(line) > LONG_LINE else [line])
        return out
    return [s for para in sec.text.split("\n") for s in split_sentences(para)]


def pack(units: list[str], target: int, overlap_units: int, sep: str) -> list[str]:
    """단위(문장·행)를 target 길이까지 모으고, 다음 청크는 앞 청크 마지막 overlap_units개로 시작한다."""
    chunks, cur, size = [], [], 0
    for u in units:
        while len(u) > MAX_CHUNK:                    # 비정상적으로 긴 단위는 강제 분할
            units_head, u = u[:MAX_CHUNK], u[MAX_CHUNK:]
            if cur:
                chunks.append(sep.join(cur))
            chunks.append(units_head)
            cur, size = [], 0
        # target을 넘으면 자르되, 지금 청크가 너무 짧으면(제목 줄 하나 등) 다음 단위까지 붙인다
        if cur and size + len(u) > target and size >= target * 0.4:
            chunks.append(sep.join(cur))
            cur = cur[-overlap_units:] if overlap_units else []
            size = sum(len(x) for x in cur)
        cur.append(u)
        size += len(u) + len(sep)
    if cur and (not chunks or sep.join(cur) != chunks[-1]):
        chunks.append(sep.join(cur))
    # 짧은 꼬리 병합
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + sep + tail
    return chunks


def window(text: str, size: int, overlap: int) -> list[str]:
    """고정 길이 창. 창 끝 20% 안에 문장 끝·줄바꿈이 있으면 거기서 자른다."""
    out, i = [], 0
    while i < len(text):
        end = min(len(text), i + size)
        if end < len(text):
            zone = text[i + int(size * 0.8):end]
            m = list(re.finditer(r"[.?!다]\s|\n", zone))
            if m:
                end = i + int(size * 0.8) + m[-1].end()
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    if len(out) > 1 and len(out[-1]) < MIN_CHUNK:
        tail = out.pop()
        out[-1] = out[-1] + " " + tail
    return out


# --------------------------------------------------------------------------- #
# 문서 → 청크
# --------------------------------------------------------------------------- #
def header(doc: dict, section: str | None) -> str:
    date = (doc["published_at"] or "")[:10]
    label = SOURCE_LABEL.get(doc["source_type"], doc["source_type"])
    title = re.sub(r"^(\[(보고서|특허|공시|뉴스)\]\s*)+", "", doc["title"])   # 수집 단계에서 붙인 종류 태그 중복 제거
    lines = [f"[{label}] {title}",
             f"{doc.get('publisher') or ''} · {date}".strip(" ·")]
    if section and section not in ("본문",):
        lines.append(f"섹션: {section}")
    return "\n".join(lines)


def split_doc(doc: dict, parsed: ParsedDoc, config: str) -> list[tuple[str, str, str]]:
    """반환: [(section, content_kind, body_text)]"""
    if config == "fixed":
        full = "\n".join(s.text for s in parsed.sections)
        return [("", "text", c) for c in window(full, FIXED_SIZE, FIXED_OVERLAP)]
    prof = STRUCTURE.get(doc["source_type"], STRUCTURE["news"])
    out = []
    for sec in parsed.sections:
        if doc["source_type"] == "patent" and sec.name == "서지·요약":
            out.append((sec.name, "text", sec.text))      # 항목 하나 = 청크 하나
            continue
        sep = "\n" if (sec.kind == "table" or prof.unit == "line") else " "
        for c in pack(_units(sec, prof.unit), prof.target, prof.overlap_units, sep):
            out.append((sec.name, sec.kind, c))
    return out


def chunk_doc(doc: dict, config: str) -> list[dict]:
    parsed = parse(doc["doc_id"], doc["source_type"], doc["title"], doc["body"])
    pieces = split_doc(doc, parsed, config)
    ts = pd.to_datetime(doc["published_at"], errors="coerce")
    meta = json.loads(doc["meta"]) if isinstance(doc.get("meta"), str) and doc["meta"] else {}
    company = meta.get("corp_name") or (doc["publisher"] if doc["source_type"] in ("disclosure", "patent") else "")
    category = meta.get("category") or meta.get("board") or meta.get("country") or meta.get("ministry") or ""
    rows = []
    for i, (section, kind, body) in enumerate(pieces):
        if config == "structure_notitle":
            text = body
        elif config == "fixed":
            text = f"{header(doc, None)}\n{body}"
        else:
            text = f"{header(doc, section)}\n{body}"
        rows.append({
            "chunk_id": f"{doc['doc_id']}#{i}",
            "doc_id": doc["doc_id"],
            "chunk_index": i,
            "n_chunks": len(pieces),
            "source_type": doc["source_type"],
            "evidence_role": doc["evidence_role"],
            "origin": doc["origin"],
            "title": doc["title"],
            "publisher": doc["publisher"] or "",
            "published_at": ts.isoformat() if pd.notna(ts) else "",
            "year": int(ts.year) if pd.notna(ts) else 0,
            "quarter": f"{ts.year}Q{ts.quarter}" if pd.notna(ts) else "",
            "url": doc["url"] or "",
            "section": section,
            "content_kind": kind,
            "topics": doc["topics"] or "",
            "is_outlook": int(doc["is_outlook"] or 0),
            "event_id": doc["event_id"] if isinstance(doc.get("event_id"), str) else "",
            "company": company or "",
            "category": str(category),
            "char_len": len(body),
            "config": config,
            "text": text,          # 임베딩·검색 대상
            "body": body,          # 답변 생성 시 인용용 (머리말 제외)
            "text_hash": hashlib.sha1(text.encode()).hexdigest(),
        })
    return rows
