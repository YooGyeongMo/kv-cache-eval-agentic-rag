"""하이브리드 검색기.

조밀 검색은 오픈소스 임베딩으로, 희소 검색은 BM25로 한다.
둘을 같이 쓰는 이유는 이 문서 풀의 성격 때문이다. 평가해야 할 근거가
CXL, MLA, PIM 같은 약어와 1.80배, 93.3퍼센트 같은 수치에 몰려 있는데
조밀 검색은 이런 토큰을 주변 문장에 녹여 버리고, BM25는 철자가 같으면
반드시 집어 온다. 반대로 의미가 같고 표현이 다른 질의는 BM25가 놓친다.

가중합은 점수 비율이 아니라 RRF로 섞는다. 앞선 실험에서 EnsembleRetriever의
weights가 점수 비율이 아니라 가중 RRF라는 것을 확인했고, 정규화가 필요 없는
쪽이 문서 수가 달라져도 덜 흔들렸다.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index"

TOKEN = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?|[가-힣]+")


def tokenize(text: str) -> list[str]:
    """영문 논문과 한국어 질의를 한 토크나이저로 다룬다.

    하이픈 복합어(KV-cache)를 붙여 두고 숫자를 살린다. 한국어는 어절을
    그대로 두는데, 조사가 붙어 재현율이 떨어지는 대신 문서 쪽이 영문이라
    BM25가 실제로 일하는 구간은 약어와 숫자다.
    """
    return [t.lower() for t in TOKEN.findall(text)]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank_dense: int | None
    rank_sparse: int | None


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder, k_rrf: int = 60):
        self.chunks = chunks
        self.embedder = embedder
        self.k_rrf = k_rrf
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.matrix: np.ndarray | None = None

    # ---------- 색인 ----------
    def build(self) -> None:
        vecs = self.embedder.encode_documents([c.text for c in self.chunks])
        self.matrix = _l2_normalize(np.asarray(vecs, dtype=np.float32))

    def save(self, name: str) -> None:
        INDEX.mkdir(exist_ok=True)
        with open(INDEX / f"{name}.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, name: str, embedder) -> "HybridRetriever":
        with open(INDEX / f"{name}.pkl", "rb") as f:
            blob = pickle.load(f)
        obj = cls(blob["chunks"], embedder)
        obj.matrix = blob["matrix"]
        return obj

    # ---------- 검색 ----------
    def dense(self, query: str, k: int) -> list[int]:
        q = _l2_normalize(np.asarray([self.embedder.encode_query(query)], dtype=np.float32))
        sims = (self.matrix @ q[0])
        return list(np.argsort(-sims)[:k])

    def sparse(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:k])

    def search(self, query: str, k: int = 6, pool: int = 30,
               where: dict | None = None) -> list[Hit]:
        """RRF로 두 순위를 합친다. where로 진영이나 문서 종류를 제한한다."""
        d = self.dense(query, pool)
        s = self.sparse(query, pool)
        rank_d = {idx: r for r, idx in enumerate(d)}
        rank_s = {idx: r for r, idx in enumerate(s)}

        fused: dict[int, float] = {}
        for idx in set(d) | set(s):
            score = 0.0
            if idx in rank_d:
                score += 1.0 / (self.k_rrf + rank_d[idx] + 1)
            if idx in rank_s:
                score += 1.0 / (self.k_rrf + rank_s[idx] + 1)
            fused[idx] = score

        order = sorted(fused, key=lambda i: -fused[i])
        hits: list[Hit] = []
        for idx in order:
            c = self.chunks[idx]
            if where and any(getattr(c, key) != val for key, val in where.items()):
                continue
            hits.append(Hit(c, fused[idx], rank_d.get(idx), rank_s.get(idx)))
            if len(hits) >= k:
                break
        return hits


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(n, 1e-12, None)


def format_context(hits: list[Hit], limit: int = 7000) -> str:
    """근거를 인용 꼬리표와 함께 묶는다.

    꼬리표를 본문에 붙여 두는 이유는 생성 단계에서 문장마다 출처를 달게
    강제하기 위해서다. 논문 청크는 쪽수를 달고, 웹에서 모은 시장 문서는
    쪽수를 달지 않는다. 앞선 실험에서 웹 근거에 문서 쪽수가 잘못
    따라붙는 것을 확인했기 때문에 형식 단계에서 갈라 둔다.
    """
    out, used = [], 0
    for h in hits:
        block = f"[{h.chunk.cite}] ({h.chunk.section or h.chunk.doc_type})\n{h.chunk.text}"
        if used + len(block) > limit:
            break
        out.append(block)
        used += len(block)
    return "\n\n---\n\n".join(out)
