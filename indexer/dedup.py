"""② 인덱싱 계층 — 청크 단위 중복 처리.

문서 단위 중복(뉴스 재배포본·동일 사건)은 지식베이스 선별 전에 collector/dedup_news.py가 처리했다
(선별이 그 결과에 의존하므로 수집 계층에 둠). 여기서는 청크로 나눈 뒤에야 보이는 중복을 지운다.

  1. 완전 동일 청크   정규화한 본문이 같은 청크 → 1개만 유지
                      (정정 공시의 정정 전·후 전문, BioIN에 재게재된 KOTRA 원문과 기사 인용 등)
  2. 반복 상용구      서로 다른 문서 BOILERPLATE_MIN_DOCS개 이상에 똑같이 나오는 청크
                      (공시 표준 문구, 보고서 면책 문구 등) → 모두 제외

유지할 청크 우선순위: 1차 자료(공시·정책·특허·보고서) > 뉴스, 같은 종류면 먼저 발행된 문서.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

PRIMARY_FIRST = {"disclosure": 0, "policy": 0, "patent": 0, "report": 1, "news": 2}
BOILERPLATE_MIN_DOCS = 5
MIN_BODY_CHARS = 40


def _key(body: str) -> str:
    return hashlib.sha1(re.sub(r"[^가-힣A-Za-z0-9]", "", body).encode()).hexdigest()


def dedup_chunks(chunks: list[dict]) -> tuple[list[dict], dict]:
    """반환: (남은 청크, 통계). 통계의 doc_alias는 평가에서 '같은 내용을 가진 문서'를 정답으로 인정할 때 쓴다."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        groups[_key(c["body"])].append(i)
    drop: dict[int, str] = {}
    alias: dict[str, set[str]] = defaultdict(set)   # 제외된 청크의 문서 → 같은 내용을 남긴 문서
    for idx in groups.values():
        docs = {chunks[i]["doc_id"] for i in idx}
        if len(docs) >= BOILERPLATE_MIN_DOCS:
            for i in idx:
                drop[i] = "boilerplate"
            continue
        if len(idx) > 1:
            keep = min(idx, key=lambda i: (PRIMARY_FIRST.get(chunks[i]["source_type"], 3), chunks[i]["published_at"], i))
            for i in idx:
                if i != keep:
                    drop[i] = "duplicate"
                    if chunks[i]["doc_id"] != chunks[keep]["doc_id"]:
                        alias[chunks[i]["doc_id"]].add(chunks[keep]["doc_id"])
    for i, c in enumerate(chunks):
        if i not in drop and len(re.sub(r"\s", "", c["body"])) < MIN_BODY_CHARS:
            drop[i] = "too_short"
    kept = [c for i, c in enumerate(chunks) if i not in drop]
    # 문서별 청크 번호·개수 다시 매김
    per_doc: dict[str, list[dict]] = defaultdict(list)
    for c in kept:
        per_doc[c["doc_id"]].append(c)
    for doc_chunks in per_doc.values():
        for j, c in enumerate(doc_chunks):
            c["chunk_index"], c["n_chunks"] = j, len(doc_chunks)
            c["chunk_id"] = f"{c['doc_id']}#{j}"
    kept_docs = {c["doc_id"] for c in kept}
    stats = {"input": len(chunks), "kept": len(kept),
             "dropped": {r: sum(1 for x in drop.values() if x == r) for r in ("duplicate", "boilerplate", "too_short")},
             "docs_fully_deduplicated": len({c["doc_id"] for c in chunks} - kept_docs),
             "doc_alias": {d: sorted(v) for d, v in alias.items()}}
    return kept, stats
