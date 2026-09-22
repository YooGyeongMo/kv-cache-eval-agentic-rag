"""오픈소스 임베딩 래퍼.

과제가 오픈소스 임베딩을 요구한다. 어떤 모델을 쓸지는 리더보드 순위가 아니라
이 문서 풀에서 실측한 Hit Rate와 MRR로 고른다(eval/embed_bench.py).

모델마다 접두사 규약이 다르다. E5 계열은 query와 passage를 구분해 붙여야
성능이 나오고, BGE 계열은 질의 쪽에만 지시문을 붙인다. 이것을 맞추지 않으면
모델 잘못이 아니라 사용법 잘못으로 점수가 떨어져서 비교가 의미를 잃는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmbedSpec:
    name: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq: int = 512
    note: str = ""
    params_m: int = 0


CANDIDATES: dict[str, EmbedSpec] = {
    "bge-m3": EmbedSpec(
        name="bge-m3", hf_id="BAAI/bge-m3", max_seq=1024, params_m=568,
        note="다국어, 긴 입력 허용. 한국어 질의와 영문 논문을 한 공간에 둔다.",
    ),
    "me5-large": EmbedSpec(
        name="me5-large", hf_id="intfloat/multilingual-e5-large",
        query_prefix="query: ", doc_prefix="passage: ", max_seq=512, params_m=560,
        note="다국어 대조학습 계열의 기준점. 접두사 규약이 엄격하다.",
    ),
    "kure-v1": EmbedSpec(
        name="kure-v1", hf_id="nlpai-lab/KURE-v1", max_seq=1024, params_m=568,
        note="한국어 검색에 맞춰 추가 학습한 bge-m3 계열. 질의가 한국어라 후보에 넣는다.",
    ),
    "mini-multi": EmbedSpec(
        name="mini-multi", hf_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_seq=128, params_m=118,
        note="경량 대조군. 비용을 5분의 1로 줄였을 때 검색이 얼마나 망가지는지 본다.",
    ),
}


class Embedder:
    """sentence-transformers 모델을 접두사 규약과 함께 감싼다."""

    def __init__(self, spec: EmbedSpec, device: str | None = None, batch: int = 8):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.batch = batch
        self.model = SentenceTransformer(spec.hf_id, device=device)
        self.model.max_seq_length = spec.max_seq

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        payload = [self.spec.doc_prefix + t for t in texts]
        return self.model.encode(payload, batch_size=self.batch,
                                 show_progress_bar=True, normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.spec.query_prefix + text,
                                 normalize_embeddings=True)


def load(name: str, **kw) -> Embedder:
    if name not in CANDIDATES:
        raise KeyError(f"모르는 임베딩 후보: {name}. 가능한 값: {list(CANDIDATES)}")
    return Embedder(CANDIDATES[name], **kw)
