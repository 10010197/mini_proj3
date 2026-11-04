"""② 인덱싱 계층 — 임베딩 생성 (OpenAI Embeddings API) + 캐시.

모델 결정: text-embedding-3-small (1536차원)  — 근거는 docs/indexing.md §1
  - 한국어를 포함한 다국어 검색 성능이 검증된 범용 모델, 로컬 GPU 없이 사용 가능
  - 비용 $0.02 / 100만 토큰: 지식베이스 전체(청크 약 4만 개)를 청킹 설정마다 다시 임베딩해도 1회 약 $0.5
  - 3-large(3072차원)는 6.5배 비용 → 같은 청킹에서 Hit@3 차이를 측정해 판단 (eval/retrieval_eval.py --model)

캐시: store/embeddings/{model}.sqlite  (text_hash → float32 벡터). 같은 텍스트는 다시 요청하지 않는다.
※ 챗봇(④)은 이 모듈로 문서를 임베딩하지 않는다. 질의 임베딩만 rag/retriever.py에서 같은 모델로 만든다.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

from collector import storage

log = logging.getLogger("indexer.embed")

DEFAULT_MODEL = "text-embedding-3-small"
EMB_DIR = storage.PROJECT_ROOT / "store" / "embeddings"
BATCH = 128
TOKENS_PER_MIN = 900_000          # 계정 한도 1M TPM보다 약간 낮게
MAX_INPUT_CHARS = 6000            # 8,191토큰 한도 안전선 (한국어 약 1자≈1토큰)


def _client():
    import openai

    key = dotenv_values(storage.PROJECT_ROOT / ".env").get("OPENAI_API_KEY")
    return openai.OpenAI(api_key=key)


def _db(model: str) -> sqlite3.Connection:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(EMB_DIR / f"{model}.sqlite")
    con.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")
    return con


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL, normalize: bool = True) -> np.ndarray:
    hashes = [text_hash(t) for t in texts]
    con = _db(model)
    have: dict[str, np.ndarray] = {}
    uniq = list(dict.fromkeys(hashes))
    for i in range(0, len(uniq), 900):
        part = uniq[i:i + 900]
        for h, v in con.execute(f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(part))})", part):
            have[h] = np.frombuffer(v, dtype=np.float32)
    todo = {h: t for h, t in zip(hashes, texts) if h not in have}
    if todo:
        log.info("%s: 캐시 %d, 새로 임베딩 %d", model, len(have), len(todo))
        client = _client()
        items = list(todo.items())
        window_start, used = time.monotonic(), 0
        for b in range(0, len(items), BATCH):
            batch = items[b:b + BATCH]
            est = sum(len(t) for _, t in batch)
            if used + est > TOKENS_PER_MIN:
                wait = 60 - (time.monotonic() - window_start)
                if wait > 0:
                    time.sleep(wait)
                window_start, used = time.monotonic(), 0
            for attempt in range(6):
                try:
                    resp = client.embeddings.create(model=model, input=[t[:MAX_INPUT_CHARS] for _, t in batch])
                    break
                except Exception as e:  # 429·일시 오류 → 지수 대기
                    if attempt == 5:
                        raise
                    log.warning("재시도 %d: %s", attempt + 1, str(e)[:120])
                    time.sleep(5 * 2 ** attempt)
            used += resp.usage.total_tokens
            rows = []
            for (h, _), d in zip(batch, resp.data):
                v = np.asarray(d.embedding, dtype=np.float32)
                have[h] = v
                rows.append((h, v.tobytes()))
            con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            con.commit()
            if (b // BATCH) % 20 == 0:
                log.info("  %d/%d", min(b + BATCH, len(items)), len(items))
    con.close()
    mat = np.vstack([have[h] for h in hashes]) if hashes else np.zeros((0, 1), np.float32)
    if normalize:
        mat = mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
    return mat
