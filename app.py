"""실행 스크립트.

  python app.py index      문서 풀을 읽어 색인을 만든다. 한 번만 하면 된다.
  python app.py run        평가를 돌리고 보고서를 만든다.
  python app.py run --seed 1   기술 제시 순서를 뒤집어 다시 돌린다.
  python app.py compare    두 순서의 결과가 갈리는지 비교한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import config
import llm
from graph import build
from rag import embedder as emb
from rag.ingest import build_pool, pool_page_count
from rag.retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
POOL = ROOT / "data" / "pool.json"
INDEX_NAME = "pool"

PAGE_LIMIT = 200


def load_spec() -> dict:
    return json.loads(POOL.read_text(encoding="utf-8"))


def cmd_index() -> None:
    spec = load_spec()
    pages = pool_page_count(spec)
    if pages > PAGE_LIMIT:
        sys.exit(f"문서 풀이 {pages}쪽이다. 과제 제한 {PAGE_LIMIT}쪽을 넘는다.")
    chunks = build_pool(spec)
    print(f"문서 풀 {pages}쪽, 청크 {len(chunks)}개, 임베딩 {config.EMBEDDING}")
    r = HybridRetriever(chunks, emb.load(config.EMBEDDING))
    t0 = time.time()
    r.build()
    r.save(INDEX_NAME)
    print(f"색인 완료 {time.time() - t0:.1f}초 -> index/{INDEX_NAME}.pkl")


def make_web_search():
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=5, search_depth="advanced")

    def search(query: str) -> list[dict]:
        try:
            res = tool.invoke({"query": query})
        except Exception as e:
            print(f"  [웹 검색 실패] {type(e).__name__}: {e}")
            return []
        items = res.get("results", []) if isinstance(res, dict) else []
        return [{"title": i.get("title", ""), "url": i.get("url", ""),
                 "content": i.get("content", "")} for i in items]

    return search


def setup_note() -> str:
    bench = OUT / "embed_bench.json"
    if not bench.exists():
        return "임베딩 비교 실측 결과가 없다."
    rows = json.loads(bench.read_text(encoding="utf-8"))
    lines = ["임베딩 후보를 이 문서 풀에서 직접 재고 골랐다. 질의 20개, 정답은 사람이 고른 문자열이다.",
             "| 모델 | 파라미터 | 색인 초 | 한국어 질의 Hit@5 | 영어 질의 Hit@5 | 영어 질의 MRR |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['params_m']}M | {r['index_seconds']} | "
                     f"{r['ko']['dense_hit@5']} | {r['en']['hybrid_hit@5']} | {r['en']['hybrid_mrr']} |")
    lines.append(f"최종 선택은 {config.EMBEDDING}이다.")
    return "\n".join(lines)


def limits_note(state: dict) -> str:
    parts = [
        "이 평가는 1인 과제 범위로 한정했다. 네 관점 중 기술 성숙도(TRL) 관점과 "
        "이해관계자 관점은 범위에서 뺐다. 빠뜨린 것이 아니라 뺀 것이다.",
        "두 기술의 공개 시점이 2024년 6월과 2026년 6월로 2년 차이 난다. "
        "채택 현황을 나란히 놓으면 늦게 나온 쪽이 불리하게 읽힌다. 이 시간 차이를 "
        "감안하고 읽어야 한다.",
        "시장 근거는 공개 자료만 썼다. 반도체 업계는 수율과 원가를 공개하지 않으므로 "
        "실제 채택 규모는 공개 정보로 확인할 수 없다.",
        "웹 근거에 출처 등급을 붙였다. 일차는 논문과 코드 저장소와 벤더 공식 문서, "
        "시장조사는 상업 조사기관 요약, 이차는 그 밖의 기사와 블로그다. 시장 규모 "
        "수치는 대부분 시장조사 등급이며 원본 보고서를 열어 대조하지 못했다. 같은 "
        "범주를 두고도 조사기관마다 수치가 크게 다른 경우가 있으므로 단일 수치를 "
        "그대로 믿으면 안 된다.",
    ]
    audit = state.get("audit", {})
    tiers = {t: r for t, r in (audit.get("per_tech") or {}).items()}
    if tiers:
        parts.append("수집된 웹 근거의 등급 분포는 다음과 같다. " + " ".join(
            f"{t}: 일차 {r.get('web_일차', 0)}건, 이차 {r.get('web_이차', 0)}건, "
            f"시장조사 {r.get('web_시장조사', 0)}건." for t, r in tiers.items()))
    if audit.get("flags"):
        parts.append("확증편향 감사에서 다음이 걸렸다. " + " ".join(audit["flags"]))
    else:
        parts.append("확증편향 감사에서 한쪽으로 기운 항목은 나오지 않았다.")
    parts.append("취한 조치는 다음과 같다. " + " ".join(
        f"{k}: {v}" for k, v in config.BIAS_GUARDS.items()))
    return "\n".join(parts)


def cmd_run(seed: int) -> None:
    spec = load_spec()
    chunks = build_pool(spec)
    r = HybridRetriever.load(INDEX_NAME, emb.load(config.EMBEDDING))
    r.chunks = chunks
    app = build(r, make_web_search(), setup_note(), limits_note)

    init = {
        "techs": config.TECHS,
        "domain": config.PRIMARY_DOMAIN,
        "contrast_domains": config.CONTRAST_DOMAINS,
        "order_seed": seed,
        "evidence": [],
    }
    t0 = time.time()
    state = app.invoke(init, {"recursion_limit": 60})

    OUT.mkdir(exist_ok=True)
    tag = f"seed{seed}"
    (OUT / f"report_{tag}.md").write_text(state["report_md"], encoding="utf-8")
    (OUT / f"state_{tag}.json").write_text(
        json.dumps({k: v for k, v in state.items() if k != "techs"},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n보고서 outputs/report_{tag}.md")
    print(f"LLM 호출 {llm.USAGE['calls']}회, 입력 {llm.USAGE['in']:,} 출력 {llm.USAGE['out']:,} 토큰")
    print(f"소요 {time.time() - t0:.0f}초")
    print(f"감사: {'통과' if state['audit']['passed'] else '지적 ' + str(len(state['audit']['flags'])) + '건'}")
    v = state.get("verify", {})
    print(f"검증: {'통과' if v.get('passed') else '지적 ' + str(len(v.get('problems', [])))+'건'}")
    for p in v.get("problems", []):
        print("   -", p)


def cmd_compare() -> None:
    """순서를 바꿔 두 번 돌린 결과가 갈리는지 본다."""
    a = OUT / "state_seed0.json"
    b = OUT / "state_seed1.json"
    if not (a.exists() and b.exists()):
        sys.exit("먼저 run --seed 0 과 run --seed 1 을 각각 돌린다.")
    sa, sb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    print("순서 효과 점검")
    for key in ("market", "domain_fit"):
        for tid in sa.get(key, {}):
            va = sa[key].get(tid, {})
            vb = sb.get(key, {}).get(tid, {})
            print(f"  {key:<11}{tid:<6} 유리 {len(va.get('pro', []))} 대 {len(vb.get('pro', []))}"
                  f"   불리 {len(va.get('con', []))} 대 {len(vb.get('con', []))}"
                  f"   근거없음 {len(va.get('unknown', []))} 대 {len(vb.get('unknown', []))}")
    ca, cb = len(sa.get("conflicts", [])), len(sb.get("conflicts", []))
    print(f"  상충 지점 수 {ca} 대 {cb}")


def cmd_final() -> None:
    """제출물을 처음부터 끝까지 만든다.

    색인, 두 방향 실행(순서 효과 확인용), 비교, PDF 변환, 최종 점검을
    차례로 돌린다. 어느 단계에서 막히면 거기서 멈추고 이유를 알린다.
    """
    import subprocess

    py = sys.executable
    sub = config.SUBMISSION
    steps = [
        ("문서 풀 색인", [py, "app.py", "index"]),
        ("평가 실행 (정방향)", [py, "app.py", "run", "--seed", "0"]),
        ("평가 실행 (역방향, 순서 효과 확인)", [py, "app.py", "run", "--seed", "1"]),
        ("순서 효과 비교", [py, "app.py", "compare"]),
        ("PDF 변환", [py, "tools/md2pdf.py", "outputs/report_seed0.md",
                      f"KV cache 최적화 기술 다관점 평가 · {sub['campus']} {sub['class']} {sub['author']}"]),
    ]
    for label, cmd in steps:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"\n[중단] '{label}' 단계가 실패했다. 위 오류를 확인한다.")

    # 제출 파일명으로 복사한다
    src = OUT / "report_seed0.pdf"
    dest = OUT / f"RAG-Output_{sub['campus']}_{sub['class']}_{sub['author']}.pdf"
    if src.exists():
        dest.write_bytes(src.read_bytes())
        print(f"\n제출 파일 {dest}")

    print(f"\n{'=' * 70}\n최종 점검\n{'=' * 70}")
    subprocess.run([py, "tools/qa.py"], cwd=ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["index", "run", "compare", "final"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"index": lambda: cmd_index(),
     "run": lambda: cmd_run(args.seed),
     "compare": lambda: cmd_compare(),
     "final": lambda: cmd_final()}[args.command]()
