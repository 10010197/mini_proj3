"""④ 챗봇 계층 — BM25 키워드 검색 (읽기 전용) + 공용 토크나이저.

인덱스는 인덱서(indexer/bm25.py)가 만들어 store/bm25/{config}.npz 에 저장한다. 여기서는 읽기만 한다.
토크나이저는 색인·질의가 반드시 같아야 하므로 이 모듈 하나에 두고 인덱서가 import 한다.

토큰
  - 한글 연속 구간 → 글자 2-gram ('삼성바이오' → 삼성, 성바, 바이, 이오). 형태소 분석기 없이 조사·띄어쓰기 차이를 흡수한다.
  - 영문·숫자 → 소문자 단어. 하이픈으로 이어진 코드는 붙인 형태도 추가 ('ALT-B4' → alt, b4, altb4).
    벡터 검색이 약한 고유명사·제품 코드(LCB14, ALT-B4, SOTIO)를 정확히 맞추기 위한 것이다.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

BM25_DIR = Path(__file__).resolve().parents[1] / "store" / "bm25"
K1, B = 1.2, 0.75

HANGUL = re.compile(r"[가-힣]+")
ALNUM = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for run in HANGUL.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    for w in ALNUM.findall(text):
        w = w.lower()
        parts = w.split("-")
        toks.extend(p for p in parts if len(p) >= 2 or p.isdigit())
        if len(parts) > 1:
            toks.append("".join(parts))
    return toks


class BM25Index:
    """역색인: 단어 → (청크 번호 배열, 미리 계산한 BM25 가중치 배열). 점수 = 질의 단어 가중치의 합."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.chunk_ids = z["chunk_ids"]
        self.vocab = {t: i for i, t in enumerate(z["vocab"].tolist())}
        self.offsets, self.postings, self.weights = z["offsets"], z["postings"], z["weights"]

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.chunk_ids), dtype=np.float32)
        for t in set(tokenize(query)):
            i = self.vocab.get(t)
            if i is None:
                continue
            s, e = self.offsets[i], self.offsets[i + 1]
            np.add.at(scores, self.postings[s:e], self.weights[s:e])
        if not scores.any():
            return []
        k = min(k, int((scores > 0).sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(str(self.chunk_ids[i]), float(scores[i])) for i in top]


@lru_cache(maxsize=4)
def load(config: str) -> BM25Index:
    path = BM25_DIR / f"{config}.npz"
    if not path.exists():
        raise RuntimeError(f"{path}가 없습니다. 먼저 `python -m indexer bm25 --configs {config}`로 만드세요.")
    return BM25Index(path)
