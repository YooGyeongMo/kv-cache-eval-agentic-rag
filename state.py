"""그래프 State 정의.

설계 원칙 두 가지를 지킨다.

첫째, 병렬로 갱신되는 값은 키를 나눈다. 시장 평가와 도메인 평가가 fan-out으로
동시에 돌기 때문에 한 키에 같이 쓰면 마지막에 쓴 쪽이 이긴다. 관점마다 키를
따로 두고, 그 안에서도 기술별로 갈라 둔다.

둘째, 근거는 한 원장에 모은다. 평가 문장과 근거를 따로 들고 다니면 보고서
단계에서 짝이 어긋난다. evidence를 덧붙이기 전용 목록으로 두고, 모든 평가
문장은 이 원장의 id를 가리키게 한다. 출처를 못 대는 문장은 검증 단계에서
떨어진다.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


def merge_by_key(left: dict, right: dict) -> dict:
    """기술 id를 키로 쓰는 사전을 병렬 갱신에서 안전하게 합친다."""
    out = dict(left or {})
    for k, v in (right or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def add_counts(left: dict, right: dict) -> dict:
    out = dict(left or {})
    for k, v in (right or {}).items():
        out[k] = out.get(k, 0) + v
    return out


class Evidence(TypedDict):
    """평가 문장이 가리키는 근거 한 줄."""
    id: str                       # E001 같은 일련번호
    tech: str                     # 어느 기술에 대한 근거인가
    perspective: str              # market 또는 domain
    stance: Literal["pro", "con", "neutral"]   # 대칭 수집을 확인하는 축
    claim: str                    # 근거가 말하는 내용
    cite: str                     # 문서 풀 인용 꼬리표 또는 웹 출처 제목
    source_type: Literal["pool", "web"]
    url: str                      # 웹 근거만 채운다


class TechBrief(TypedDict):
    tech: str
    camp: Literal["sw", "hw"]
    approach: str
    reported_numbers: list[str]
    scope: str
    limits: str
    evidence_ids: list[str]


class PerspectiveView(TypedDict):
    tech: str
    criterion_scores: dict[str, str]   # 기준별 서술. 점수가 아니라 관찰이다.
    pro: list[str]
    con: list[str]
    unknown: list[str]                 # 근거를 못 찾은 항목을 남긴다
    evidence_ids: list[str]


class EvalState(TypedDict, total=False):
    # ---- 입력. 실행 내내 바뀌지 않는다 ----
    techs: list[dict]                  # config에서 고른 2건
    domain: str                        # 주 도메인
    contrast_domains: list[str]        # 평가가 뒤집히는지 보려고 두는 대조 도메인
    order_seed: int                    # 제시 순서. 0과 1로 두 번 돌려 순서 효과를 본다

    # ---- 기술 조사. RAG 루프를 거친다 ----
    tech_briefs: Annotated[dict[str, TechBrief], merge_by_key]

    # ---- 관점별 평가. 여기부터 fan-out이라 키를 나눈다 ----
    market: Annotated[dict[str, PerspectiveView], merge_by_key]
    domain_fit: Annotated[dict[str, PerspectiveView], merge_by_key]

    # ---- 근거 원장. 모든 노드가 덧붙이기만 한다 ----
    evidence: Annotated[list[Evidence], operator.add]

    # ---- 루프 제어 ----
    retrieve_k: Annotated[dict[str, int], add_counts]   # 재검색 시 넓히는 폭
    retries: Annotated[dict[str, int], add_counts]

    # ---- 감사와 종합 ----
    audit: dict                        # 대칭성, 근거 누락, 금지 표현
    conflicts: list[dict]              # 관점이 엇갈리는 지점
    synthesis: dict

    # ---- 산출 ----
    report_md: str
    verify: dict
    n_report: int
    extra_numbers: str
