"""① 수집 계층 — 원문 저장 (store/docs.db).

모든 수집기(load_provided, crawl_dart, crawl_* …)가 같은 documents 스키마로 적재한다.
출처 종류(source_type)마다 주제 B의 '시간축 근거 역할(evidence_role)'을 갖는다.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "store" / "docs.db"

# 출처 종류 → 기본 근거 역할 (문서별로 덮어쓸 수 있음. 예: 공시 중 실적 '전망' 공시는 outlook)
EVIDENCE_ROLE = {
    "news": "current",        # 뉴스 = 현재 시점의 사건·발표
    "disclosure": "current",  # 상장사 공시 = 기업 단위 1차 근거
    "policy": "current",      # 정부 보도자료 = 정책·규제 원문
    "patent": "leading",      # 특허 = 기술 선행 지표
    "clinical": "leading",    # 임상·허가 = 상업화 선행 지표
    "report": "outlook",      # 산업·투자 보고서 = 시장 전망
}

DOCUMENT_COLUMNS = [
    "doc_id", "source_type", "evidence_role", "origin", "title", "body", "url",
    "published_at", "publisher", "collect_keywords", "topics", "is_outlook",
    "keyword_hits", "char_len", "source_files", "meta", "loaded_at",
]

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,
    source_type      TEXT NOT NULL,      -- news | disclosure | policy | patent | clinical | report
    evidence_role    TEXT NOT NULL,      -- current | leading | outlook
    origin           TEXT NOT NULL,      -- provided | dart | kipris ...
    title            TEXT NOT NULL,
    body             TEXT NOT NULL,
    url              TEXT,
    published_at     TEXT,               -- ISO 8601
    publisher        TEXT,               -- 매체명 / 공시 회사명 / 발행기관
    collect_keywords TEXT,               -- '; ' 구분
    topics           TEXT,               -- '; ' 구분
    is_outlook       INTEGER DEFAULT 0,
    keyword_hits     INTEGER,
    char_len         INTEGER,
    source_files     TEXT,
    meta             TEXT,               -- 출처별 추가 메타데이터(JSON)
    loaded_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(published_at);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_origin ON documents(origin);
CREATE TABLE IF NOT EXISTS rejected_docs (
    doc_id       TEXT PRIMARY KEY,
    origin       TEXT,
    title        TEXT,
    url          TEXT,
    published_at TEXT,
    drop_reason  TEXT,
    char_len     INTEGER,
    loaded_at    TEXT
);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(DDL)
    # 이전 버전 스키마 호환 + 지식베이스 선별 컬럼 (collector/select_kb.py가 채움)
    cols = {r[1] for r in con.execute("PRAGMA table_info(documents)")}
    for col, typ in (("meta", "TEXT"), ("in_kb", "INTEGER DEFAULT 0"), ("kb_reason", "TEXT")):
        if col not in cols:
            con.execute(f"ALTER TABLE documents ADD COLUMN {col} {typ}")
    return con


def _insert(con: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    cols = ", ".join(frame.columns)
    marks = ", ".join("?" * len(frame.columns))
    rows = frame.astype(object).where(frame.notna(), None).itertuples(index=False, name=None)
    con.executemany(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", rows)


def save_documents(
    docs: pd.DataFrame,
    origin: str,
    db_path: Path = DEFAULT_DB,
    rejected: pd.DataFrame | None = None,
    replace_origin: bool = False,
) -> None:
    """documents(및 rejected_docs)에 upsert. replace_origin=True면 해당 origin 문서를 먼저 비운다."""
    docs = docs.reindex(columns=DOCUMENT_COLUMNS)
    docs["origin"] = origin
    docs["loaded_at"] = docs["loaded_at"].fillna(now_iso())
    with connect(db_path) as con:
        if replace_origin:
            con.execute("DELETE FROM documents WHERE origin = ?", (origin,))
            con.execute("DELETE FROM rejected_docs WHERE origin = ?", (origin,))
        _insert(con, "documents", docs)
        if rejected is not None:
            rej = rejected.copy()
            rej["origin"] = origin
            _insert(con, "rejected_docs", rej)


def existing_ids(origin: str, db_path: Path = DEFAULT_DB) -> set[str]:
    if not Path(db_path).exists():
        return set()
    with connect(db_path) as con:
        return {r[0] for r in con.execute("SELECT doc_id FROM documents WHERE origin = ?", (origin,))}
