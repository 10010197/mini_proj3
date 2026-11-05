"""② 인덱싱 계층 — BM25 키워드 인덱스 생성 (하이브리드 검색용).

    store/chunks/{config}.jsonl ─▶ 토큰화(rag.bm25.tokenize, 질의와 같은 규칙) ─▶ 역색인 + BM25 가중치 ─▶ store/bm25/{config}.npz

- 색인 대상은 청크 `text`(머리말 포함). 제목·회사명·제품 코드가 머리말에 있어 키워드 매칭이 잘 된다.
- 가중치는 미리 계산해 둔다: idf × tf·(k1+1) / (tf + k1·(1−b+b·len/avglen)). 질의 시점에는 더하기만 한다.
- 챗봇은 이 파일을 읽기만 한다(rag/bm25.py).
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

import numpy as np

from rag.bm25 import B, BM25_DIR, K1, BM25Index, tokenize

log = logging.getLogger("indexer.bm25")


def build(config: str) -> dict:
    from indexer.__main__ import read_chunks

    chunks = read_chunks(config)
    post: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lens = np.zeros(len(chunks), dtype=np.float32)
    for i, c in enumerate(chunks):
        tf = Counter(tokenize(c["text"]))
        lens[i] = sum(tf.values())
        for t, n in tf.items():
            post[t].append((i, n))
    avg = float(lens.mean()) if len(lens) else 1.0
    n_docs = len(chunks)
    vocab = sorted(post)
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    postings, weights = [], []
    for j, t in enumerate(vocab):
        plist = post[t]
        idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
        ids = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
        tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
        w = idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * lens[ids] / avg))
        postings.append(ids)
        weights.append(w.astype(np.float32))
        offsets[j + 1] = offsets[j] + len(plist)
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    path = BM25_DIR / f"{config}.npz"
    np.savez_compressed(path, chunk_ids=np.array([c["chunk_id"] for c in chunks]), vocab=np.array(vocab),
                        offsets=offsets, postings=np.concatenate(postings), weights=np.concatenate(weights))
    # 검증: 다시 읽어 청크 수 일치 + 자기 제목으로 검색하면 자기 청크가 상위에 오는지(표본 50개)
    idx = BM25Index(path)
    rnd = np.random.default_rng(42)
    sample = [chunks[i] for i in rnd.choice(len(chunks), size=min(50, len(chunks)), replace=False)]
    self_hit = sum(1 for c in sample if c["chunk_id"] in {cid for cid, _ in idx.search(c["text"][:200], 10)})
    out = {"config": config, "chunks": n_docs, "indexed": len(idx), "vocab": len(vocab),
           "postings": int(offsets[-1]), "avg_tokens": round(avg, 1), "self_hit@10": f"{self_hit}/{len(sample)}",
           "ok": len(idx) == n_docs and self_hit >= 0.9 * len(sample)}
    log.info("BM25 %s: %s", config, out)
    return out
