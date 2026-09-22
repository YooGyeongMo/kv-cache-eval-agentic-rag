"""그래프 조립.

흐름은 순차, 병렬, 반복을 모두 쓴다.

  준비 → 기술 조사(기술별 병렬) → 관점 평가(기술 곱하기 관점 병렬)
       → 감사 → (기울었으면 되돌아가 보강) → 종합 → 보고서 → 검증
       → (통과 못 하면 보고서 다시) → 끝

병렬 구간이 둘 있다. 기술 조사는 기술 2건이 서로를 보지 않고 도는 편이
낫다. 한쪽 결과를 보고 다른 쪽을 조사하면 먼저 본 쪽이 기준이 된다.
관점 평가도 마찬가지로 시장성과 도메인이 서로를 참조하지 않는다. 겹치는
내용은 종합 단계에서 잇는다.

되돌아가는 길도 둘이다. 감사가 한쪽 근거만 모였다고 판정하면 모자란 쪽을
다시 모으고, 검증이 막히면 보고서를 다시 쓴다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import config
from agents import nodes
from agents.retrieve_loop import make_subgraph
from state import EvalState

MAX_RECOLLECT = 1


def build(retriever, web_search, setup_note: str, limits_note: str):
    sub = make_subgraph(retriever, web_search)

    survey = nodes.make_survey(sub)
    market = nodes.make_perspective(sub, "market")
    domain = nodes.make_perspective(sub, "domain")
    report = nodes.make_report(setup_note, limits_note)

    def prepare(s: EvalState) -> dict:
        techs = list(s["techs"])
        if s.get("order_seed", 0) % 2 == 1:
            techs = list(reversed(techs))     # 순서 효과를 보려고 뒤집는다
        return {"techs": techs, "n_report": 0, "retries": {}}

    def fan_survey(s: EvalState):
        return [Send("survey", {"tech": t}) for t in s["techs"]]

    def fan_perspective(s: EvalState):
        out = []
        for t in s["techs"]:
            out.append(Send("market", {"tech": t, "domain": s["domain"]}))
            out.append(Send("domain", {"tech": t, "domain": s["domain"],
                                       "contrast": s.get("contrast_domains", [])}))
        return out

    def after_audit(s: EvalState) -> str:
        if s["audit"]["passed"]:
            return "ok"
        if s.get("retries", {}).get("audit", 0) >= MAX_RECOLLECT:
            return "ok"        # 기운 채로 넘어가되 그 사실을 보고서 한계 장에 적는다
        return "recollect"

    def recollect(s: EvalState) -> dict:
        """모자란 쪽을 한 번 더 모은다. 무한히 돌지 않는다."""
        return {"retries": {"audit": 1}}

    def after_verify(s: EvalState) -> str:
        if s["verify"]["passed"] or s.get("n_report", 0) >= nodes.MAX_REPORT_TRY:
            return "done"
        return "rewrite"

    g = StateGraph(EvalState)
    g.add_node("prepare", prepare)
    g.add_node("survey", survey)
    g.add_node("collect_join", lambda s: {})
    g.add_node("market", market)
    g.add_node("domain", domain)
    g.add_node("audit", nodes.audit)
    g.add_node("recollect", recollect)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("report", report)
    g.add_node("verify", nodes.verify)

    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", fan_survey, ["survey"])
    g.add_edge("survey", "collect_join")
    g.add_conditional_edges("collect_join", fan_perspective, ["market", "domain"])
    g.add_edge("market", "audit")
    g.add_edge("domain", "audit")
    g.add_conditional_edges("audit", after_audit,
                            {"ok": "synthesize", "recollect": "recollect"})
    g.add_conditional_edges("recollect", fan_perspective, ["market", "domain"])
    g.add_edge("synthesize", "report")
    g.add_edge("report", "verify")
    g.add_conditional_edges("verify", after_verify, {"done": END, "rewrite": "report"})
    return g.compile()
