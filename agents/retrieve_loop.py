"""근거 수집 서브그래프.

세 에이전트가 모두 이 서브그래프를 불러 쓴다. 검색을 한 번 하고 마는 대신
관련성을 판정하고, 부족하면 다시 시도하는 구조라서 Agentic RAG에 해당한다.

재시도 방식을 앞선 실험 결과에 맞춰 정했다.

첫째, 같은 질의로 다시 검색하지 않는다. 앞선 실험에서 재검색 반복이 같은
문서를 그대로 다시 가져왔고, 질의를 그대로 둔 채 k를 6에서 8로 넓혔을 때
정답이 27에서 30으로 늘었다. 그래서 1차 재시도는 질의를 건드리지 않고 k만 넓힌다.

둘째, 그래도 안 되면 질의를 영어로 다시 쓴다. 문서가 영문 논문인데 질의가
한국어라 표현이 어긋나는 경우가 이 문서 풀의 주된 실패 원인이다.

셋째, 논문에서 찾아야 하는 사실에는 웹 검색으로 넘어가지 않는다. 앞선 실험에서
웹 폴백을 열었더니 웹 질문 8개를 얻는 대신 문서 질문 오답이 1개에서 4개에서
7개까지 늘었다. 논문에 없으면 없다고 남기는 쪽이 낫다. 웹 검색은 애초에
논문에 있을 수 없는 정보, 즉 시장 채택 현황과 제품 출시 같은 항목에만 연다.

넷째, 문서 풀을 뒤질 때는 질의를 영어로 옮긴 다음 검색한다. eval/embed_bench.py
실측에서 한국어 질의는 하이브리드 검색이 오히려 점수를 깎았고(Hit@5 0.65에서
0.50), 같은 질의를 영어로 바꾸면 하이브리드가 점수를 올렸다(0.75에서 0.85).
문서가 영문이라 BM25가 한국어 어절에서는 잡음만 얹고 영어에서는 약어와
수치를 정확히 집어 오기 때문이다. 웹 검색은 한국어 자료도 걸려야 하므로
원문 질의를 그대로 쓴다.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

import prompts
from llm import ask, yes_no
from rag.retriever import HybridRetriever, format_context

MAX_WIDEN = 1
MAX_REWRITE = 1


class RetrieveState(TypedDict, total=False):
    question: str        # 사람이 읽는 질문
    query: str           # 실제로 검색에 넣는 문자열
    k: int
    where: dict | None
    allow_web: bool
    search_pool: bool    # 문서 풀을 뒤지는가. False면 웹으로 바로 간다
    n_widen: int
    n_rewrite: int
    hits: list
    web: list
    status: Literal["ok", "unknown"]
    trace: list[str]


def make_subgraph(retriever: HybridRetriever, web_search):
    """검색기와 웹 도구를 묶어 서브그래프를 만든다."""

    def localize(s: RetrieveState) -> dict:
        """문서 풀 검색에 쓸 영어 질의를 만든다. 실측 근거는 모듈 설명에 적었다."""
        if not s.get("search_pool", True):
            return {"trace": s.get("trace", []) + ["웹 전용, 원문 질의 유지"]}
        en = ask(prompts.LOCALIZE_SYS,
                 prompts.LOCALIZE_USER.format(question=s["question"])).strip()
        return {"query": en or s["question"],
                "trace": s.get("trace", []) + [f"localize -> {en[:60]}"]}

    def retrieve(s: RetrieveState) -> dict:
        hits = retriever.search(s["query"], k=s.get("k", 6), where=s.get("where"))
        return {"hits": hits,
                "trace": s.get("trace", []) + [f"retrieve k={s.get('k', 6)} q={s['query'][:60]}"]}

    def grade(s: RetrieveState) -> dict:
        if not s["hits"]:
            return {"status": "unknown"}
        ok = yes_no(prompts.RELEVANCE_SYS,
                    prompts.RELEVANCE_USER.format(
                        question=s["question"],
                        context=format_context(s["hits"], limit=4000)))
        return {"status": "ok" if ok else "unknown"}

    def widen(s: RetrieveState) -> dict:
        return {"k": s.get("k", 6) + 6, "n_widen": s.get("n_widen", 0) + 1,
                "trace": s.get("trace", []) + ["widen"]}

    def rewrite(s: RetrieveState) -> dict:
        new_q = ask(prompts.REWRITE_SYS,
                    prompts.REWRITE_USER.format(question=s["question"])).strip()
        return {"query": new_q or s["query"], "n_rewrite": s.get("n_rewrite", 0) + 1,
                "k": 6, "trace": s.get("trace", []) + [f"rewrite -> {new_q[:60]}"]}

    def web(s: RetrieveState) -> dict:
        results = web_search(s["question"])
        return {"web": results,
                "status": "ok" if results else "unknown",
                "trace": s.get("trace", []) + [f"web {len(results)}건"]}

    def route(s: RetrieveState) -> str:
        if s.get("status") == "ok":
            return "done"
        if s.get("n_widen", 0) < MAX_WIDEN:
            return "widen"
        if s.get("n_rewrite", 0) < MAX_REWRITE:
            return "rewrite"
        if s.get("allow_web"):
            return "web"
        return "done"      # 못 찾았다는 사실을 status=unknown으로 남긴 채 끝낸다

    def entry(s: RetrieveState) -> str:
        return "pool" if s.get("search_pool", True) else "web"

    g = StateGraph(RetrieveState)
    g.add_node("localize", localize)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("widen", widen)
    g.add_node("rewrite", rewrite)
    g.add_node("web", web)

    g.add_conditional_edges(START, entry, {"pool": "localize", "web": "web"})
    g.add_edge("localize", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route,
                            {"done": END, "widen": "widen",
                             "rewrite": "rewrite", "web": "web"})
    g.add_edge("widen", "retrieve")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web", END)
    return g.compile()


def collect(sub, question: str, *, k: int = 6, where: dict | None = None,
            allow_web: bool = False, search_pool: bool = True) -> RetrieveState:
    """서브그래프를 한 번 돌리고 결과를 돌려준다."""
    return sub.invoke({
        "question": question, "query": question, "k": k, "where": where,
        "allow_web": allow_web, "search_pool": search_pool,
        "n_widen": 0, "n_rewrite": 0,
        "hits": [], "web": [], "trace": [],
    })
