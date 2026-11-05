"""④ 챗봇 계층 — 검색: Chroma(store/chroma, 읽기 전용) 조회 + 문서 단위 정리 + 원문 조회.

- 인덱서가 만든 컬렉션을 **읽기만** 한다. 문서 임베딩·인덱스 쓰기·크롤링은 하지 않는다.
  질의 임베딩만 인덱서와 같은 모델(text-embedding-3-small)로 만든다.
- 청크를 넉넉히(POOL) 가져와 문서당 최대 MAX_CHUNKS_PER_DOC개만 남기고 상위 K_DOCS개 문서로 자른다.
  (같은 문서의 청크가 상위를 독차지하는 문제: docs/indexing.md §5.2)
- 원문 전체는 벡터 DB에 없다. 사용자가 원문을 요청하면 get_document(doc_id)로 store/docs.db를 조회한다.

설정(.env, 선택): EMBED_MODEL, CHUNK_CONFIG — 기본값은 인덱서와 같다.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "store" / "chroma"
DOCS_DB = PROJECT_ROOT / "store" / "docs.db"
ENV = {**dotenv_values(PROJECT_ROOT / ".env"), **{k: v for k, v in os.environ.items() if k in ("EMBED_MODEL", "CHUNK_CONFIG")}}

EMBED_MODEL = ENV.get("EMBED_MODEL") or "text-embedding-3-small"
CHUNK_CONFIG = ENV.get("CHUNK_CONFIG") or "structure"
COLLECTION = f"{CHUNK_CONFIG}__{EMBED_MODEL}"

POOL = 40                 # 1차로 가져오는 청크 수
K_DOCS = 6                # 답변 근거로 넘기는 문서 수
MAX_CHUNKS_PER_DOC = 2    # 문서당 청크 상한


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float          # 코사인 유사도 (1 - 거리)
    text: str             # 청크 text (머리말 + 본문)
    meta: dict


@lru_cache(maxsize=1)
def vectorstore():
    import chromadb
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    if not CHROMA_DIR.exists():
        raise RuntimeError("store/chroma/가 없습니다. 먼저 `python -m indexer load`로 인덱스를 적재하세요.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    names = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if COLLECTION not in names:
        raise RuntimeError(f"컬렉션 {COLLECTION}이 없습니다 (있는 것: {names}).")
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=ENV.get("OPENAI_API_KEY"))
    return Chroma(client=client, collection_name=COLLECTION, embedding_function=emb)


def search(query: str, k_docs: int = K_DOCS, max_chunks_per_doc: int = MAX_CHUNKS_PER_DOC,
           pool: int = POOL, where: dict | None = None) -> list[Hit]:
    """질의 → 상위 청크 → 문서 단위로 묶어 상위 k_docs개 문서(문서당 최대 max_chunks_per_doc 청크).
    반환 순서: 문서 순위(그 문서의 최고 점수 청크 기준) → 문서 안에서는 점수순."""
    res = vectorstore().similarity_search_with_score(query, k=pool, filter=where)
    per_doc: dict[str, list[Hit]] = {}
    order: list[str] = []
    for doc, dist in res:
        m = dict(doc.metadata)
        did = m.get("doc_id", "")
        if did not in per_doc:
            if len(order) >= k_docs:
                continue
            per_doc[did] = []
            order.append(did)
        if len(per_doc[did]) < max_chunks_per_doc:
            per_doc[did].append(Hit(m.get("chunk_id", doc.id or ""), did, round(1 - float(dist), 4), doc.page_content, m))
    return [h for d in order for h in per_doc[d]]


def get_document(doc_id: str) -> dict | None:
    """원문 조회 (store/docs.db, 읽기 전용)."""
    con = sqlite3.connect(f"file:{DOCS_DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT doc_id, source_type, evidence_role, title, body, url, published_at, publisher, loaded_at "
                          "FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()
