"""그래프 구조 고정 테스트.

README가 설명하는 흐름과 코드가 어긋나지 않게 한다. 노드를 지우거나
되돌아가는 길을 끊으면 여기서 걸린다. 평가 기준이 설계와 구현의 일치를
보므로 그 일치를 사람 눈이 아니라 테스트로 지킨다.

API 키 없이 돈다. 노드를 실행하지 않고 구조만 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                       # noqa: E402
from agents import queries as Q                     # noqa: E402
from agents.retrieve_loop import make_subgraph      # noqa: E402
from graph import build                             # noqa: E402


class _Stub:
    chunks: list = []

    def search(self, *a, **kw):
        return []


@pytest.fixture(scope="module")
def main_graph():
    return build(_Stub(), lambda q: [], "", lambda s: "")


@pytest.fixture(scope="module")
def sub_graph():
    return make_subgraph(_Stub(), lambda q: [])


def edges(g) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in g.get_graph().edges}


def nodes(g) -> set[str]:
    return set(g.get_graph().nodes)


# ---------------------------------------------------------------- 메인 그래프
def test_다섯_에이전트가_모두_있다(main_graph):
    for n in ("survey", "market", "domain", "synthesize", "report"):
        assert n in nodes(main_graph), f"{n} 노드가 없다"


def test_관점_평가가_병렬로_갈라진다(main_graph):
    e = edges(main_graph)
    assert ("collect_join", "market") in e
    assert ("collect_join", "domain") in e


def test_두_관점이_같은_감사로_모인다(main_graph):
    e = edges(main_graph)
    assert ("market", "audit") in e
    assert ("domain", "audit") in e


def test_감사가_막히면_되돌아간다(main_graph):
    """확증편향 감사가 한쪽으로 기울었다고 보면 다시 모으는 길이 있어야 한다."""
    e = edges(main_graph)
    assert ("audit", "recollect") in e
    assert ("recollect", "market") in e or ("recollect", "domain") in e


def test_검증이_막히면_보고서를_다시_쓴다(main_graph):
    assert ("verify", "report") in edges(main_graph)


def test_되돌이_상한이_있다():
    """무한 반복을 막는 상한이 설정되어 있어야 한다."""
    import graph as gmod
    from agents import nodes as nmod
    assert gmod.MAX_RECOLLECT >= 1
    assert nmod.MAX_REPORT_TRY >= 1


# ---------------------------------------------------------------- 검색 서브그래프
def test_검색_루프가_재시도_경로를_갖는다(sub_graph):
    e = edges(sub_graph)
    assert ("widen", "retrieve") in e, "k를 넓혀 다시 검색하는 길이 없다"
    assert ("rewrite", "retrieve") in e, "질의를 고쳐 다시 검색하는 길이 없다"


def test_문서_풀_검색은_현지화를_거친다(sub_graph):
    """한국어 질의를 영어로 옮긴 뒤 검색한다. 실측 근거는 모듈 설명에 있다."""
    assert ("localize", "retrieve") in edges(sub_graph)


def test_재시도_상한이_있다():
    from agents import retrieve_loop as rl
    assert rl.MAX_WIDEN >= 1
    assert rl.MAX_REWRITE >= 1


# ---------------------------------------------------------------- 설계 규약
def test_기술은_진영별로_하나씩이다():
    camps = [t["camp"] for t in config.TECHS]
    assert sorted(camps) == ["hw", "sw"], "진영별 1개씩이어야 한다"


def test_범위에서_뺀_관점은_구현하지_않는다():
    """1인 과제는 시장성과 도메인 적용 두 관점만 대상이다."""
    assert hasattr(config, "MARKET_CRITERIA")
    assert hasattr(config, "DOMAIN_CRITERIA")
    assert not hasattr(config, "TRL_CRITERIA")
    assert not hasattr(config, "STAKEHOLDER_CRITERIA")


def test_질의가_찬반_대칭이다():
    """관점마다 유리한 근거와 불리한 근거를 각각 따로 묻는다."""
    for name, qs in (("시장성", Q.MARKET), ("도메인", Q.DOMAIN)):
        stances = {q.stance for q in qs}
        assert "pro" in stances, f"{name} 질의에 유리한 쪽이 없다"
        assert "con" in stances, f"{name} 질의에 불리한 쪽이 없다"


def test_기술_조사는_웹을_쓰지_않는다():
    """논문에 있을 사실을 웹에서 찾지 않는다. 앞선 실험에서 오답이 늘었다."""
    assert all(q.web is None for q in Q.SURVEY)


def test_두_기술에_같은_질의_템플릿을_쓴다():
    a = [q.criterion for q, _, _ in Q.render(Q.MARKET, config.TECHS[0])]
    b = [q.criterion for q, _, _ in Q.render(Q.MARKET, config.TECHS[1])]
    assert a == b, "기술마다 다른 질문을 하면 비교가 성립하지 않는다"


def test_대조_도메인에_영문_대응이_있다():
    for d in config.CONTRAST_DOMAINS:
        assert d in Q.DOMAIN_EN, f"웹 검색용 영문 표현이 없다: {d}"


# ---------------------------------------------------------------- 문서와 실물의 일치
def test_README_디렉터리_구조가_실물과_맞는가():
    """구조도가 실물과 어긋나면 채점자가 코드를 못 찾는다."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    block = re.search(r"## Directory Structure\n+```\n(.*?)```",
                      (root / "README.md").read_text(encoding="utf-8"), re.S).group(1)
    # 저장소에 담기지 않는 경로. 논문은 저작권 때문에 빼고 fetch_papers.py로
    # 받으며, 산출물은 실행해야 생긴다. 새로 받은 저장소에는 없는 것이 맞다.
    GENERATED = {"data/papers", "outputs", "index"}
    cur, missing = [], []
    for line in block.split("\n"):
        if not line.strip():
            continue
        depth = len(re.match(r"^[│\s]*", line).group(0)) // 4
        name = re.sub(r"^[│├└─\s]+", "", line).split("#")[0].strip()
        if not name:
            continue
        cur = cur[:depth] + [name.rstrip("/")]
        path = "/".join(cur)
        if path not in GENERATED and not (root / path).exists():
            missing.append(path)
    assert not missing, f"구조도에 적혔으나 없는 경로: {missing}"


def test_README가_평가항목의_필수_절을_갖추었는가():
    from pathlib import Path
    t = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    for s in ("# Subject", "## Overview", "## Selected Technologies", "## Features",
              "## Tech Stack", "## Agents", "## Architecture",
              "## Directory Structure", "## Usage", "## Contributors"):
        assert s in t, f"README에 '{s}' 절이 없다"
    for k in ("Framework", "LLM/Generator", "LLM/Judge", "Retrieval", "Embedding",
              "Hit Rate@", "MRR"):
        assert k in t, f"Tech Stack에 '{k}'가 없다"
    # 과제가 Contributors에서 PM, PL 역할을 빼라고 했다
    contrib = t.split("## Contributors")[-1]
    assert "PM" not in contrib and "PL" not in contrib
