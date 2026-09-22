"""오픈소스 임베딩 후보를 이 문서 풀에서 실측해 고른다.

리더보드 순위로 고르지 않는 이유는 이 과제의 검색 조건이 리더보드와
다르기 때문이다. 문서는 영문 논문이고 질의는 한국어이며, 찾아야 하는 것은
문장의 주제가 아니라 표 안의 수치와 약어다. 이 조합은 일반 검색 벤치마크에
거의 없다.

그래서 세 가지를 같이 잰다.
  1) 한국어 질의로 영문 문서를 찾는 교차언어 검색 성능
  2) 같은 질의를 영어로 바꿨을 때의 성능. 다국어 모델이 필요한지 자체를 검증한다
  3) 청크를 벡터로 바꾸는 데 걸린 시간. 경제성이 선정 조건에 들어 있다

정답은 사람이 고른 문자열로 정한다. 그 문자열을 담은 청크가 정답 청크다.
LLM으로 정답을 만들면 채점자와 피채점자가 같은 계열이 되어 비교가 흐려진다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import embedder as emb                     # noqa: E402
from rag.ingest import build_pool                   # noqa: E402
from rag.retriever import HybridRetriever, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "eval" / "queries.json"


def gold_indices(chunks, needles: list[str]) -> list[int]:
    """정답 문자열을 모두 담은 청크를 정답으로 본다."""
    out = []
    for i, c in enumerate(chunks):
        flat = " ".join(c.text.split())
        if all(n.lower() in flat.lower() for n in needles):
            out.append(i)
    return out


def metrics(ranked: list[int], gold: set[int], k: int) -> tuple[float, float]:
    hit = 1.0 if any(i in gold for i in ranked[:k]) else 0.0
    rr = 0.0
    for r, i in enumerate(ranked, start=1):
        if i in gold:
            rr = 1.0 / r
            break
    return hit, rr


def run(name: str, chunks, cases: list[dict], k: int = 5) -> dict:
    t0 = time.time()
    e = emb.load(name)
    retr = HybridRetriever(chunks, e)
    retr.build()
    index_s = time.time() - t0

    rows = {}
    for lang in ("ko", "en"):
        hits_d = hits_h = rr_d = rr_h = 0.0
        used = 0
        for case in cases:
            gold = set(gold_indices(chunks, case["gold"]))
            if not gold:
                continue           # 정답 문자열이 풀에 없으면 채점에서 뺀다
            used += 1
            q = case[lang]
            d_rank = retr.dense(q, 30)
            h_rank = [h_.chunk_index for h_ in _with_index(retr, q, 30)]
            a, b = metrics(d_rank, gold, k)
            hits_d += a
            rr_d += b
            a, b = metrics(h_rank, gold, k)
            hits_h += a
            rr_h += b
        rows[lang] = {
            "n": used,
            "dense_hit@%d" % k: round(hits_d / max(used, 1), 3),
            "dense_mrr": round(rr_d / max(used, 1), 3),
            "hybrid_hit@%d" % k: round(hits_h / max(used, 1), 3),
            "hybrid_mrr": round(rr_h / max(used, 1), 3),
        }
    return {
        "model": name,
        "hf_id": emb.CANDIDATES[name].hf_id,
        "params_m": emb.CANDIDATES[name].params_m,
        "index_seconds": round(index_s, 1),
        "chunks": len(chunks),
        **rows,
    }


class _IdxHit:
    __slots__ = ("chunk_index",)

    def __init__(self, i):
        self.chunk_index = i


def _with_index(retr: HybridRetriever, query: str, pool: int) -> list[_IdxHit]:
    """RRF 결과를 청크 번호로 되돌린다. 채점에만 쓴다."""
    d = retr.dense(query, pool)
    s = retr.sparse(query, pool)
    rank_d = {idx: r for r, idx in enumerate(d)}
    rank_s = {idx: r for r, idx in enumerate(s)}
    fused = {}
    for idx in set(d) | set(s):
        sc = 0.0
        if idx in rank_d:
            sc += 1.0 / (retr.k_rrf + rank_d[idx] + 1)
        if idx in rank_s:
            sc += 1.0 / (retr.k_rrf + rank_s[idx] + 1)
        fused[idx] = sc
    return [_IdxHit(i) for i in sorted(fused, key=lambda i: -fused[i])]


if __name__ == "__main__":
    spec = json.loads((ROOT / "data" / "pool.json").read_text(encoding="utf-8"))
    chunks = build_pool(spec)
    cases = json.loads(QUERIES.read_text(encoding="utf-8"))

    missing = [c["id"] for c in cases if not gold_indices(chunks, c["gold"])]
    if missing:
        print(f"[경고] 정답 문자열을 문서 풀에서 못 찾은 질의: {missing}")

    names = sys.argv[1:] or list(emb.CANDIDATES)
    results = []
    for n in names:
        print(f"\n===== {n} =====")
        r = run(n, chunks, cases)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    out = ROOT / "outputs" / "embed_bench.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out}")
