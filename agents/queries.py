"""대칭 질의 묶음.

확증편향을 막는 첫 장치다. 두 기술에 같은 템플릿을 쓰고 기술 이름만 바꾼다.
관점 질의는 유리한 쪽과 불리한 쪽을 각각 따로 둔다. 한쪽만 물으면 한쪽만
모이고, 그렇게 모인 근거로 쓴 평가는 검색 설계가 만든 결론이 된다.

질의를 둘로 나눈 이유가 있다. 문서 풀 질의는 한국어로 두고 검색 직전에
영어로 옮긴다. 웹 질의는 처음부터 영어로 적는다. 1차 실행에서 한국어 웹
질의를 던졌더니 일반 IT 시장 보고서와 무관한 문서가 걸렸다.

그리고 기술 이름을 그대로 검색하지 않는다. ITME는 2026년 6월 논문이라
이름으로 찾으면 아무것도 안 나온다. 이럴 때는 그 기술이 속한 범주를
검색하고, 그 사실을 보고서에 적는다. 특정 기술의 근거가 아니라 범주의
근거라는 점을 숨기면 평가가 부풀려진다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q:
    criterion: str
    stance: str          # pro, con, neutral
    pool: str            # 문서 풀용. 한국어로 쓰고 검색 직전 영어로 옮긴다
    web: str | None      # 웹용. 영어로 적는다. None이면 웹을 쓰지 않는다


# ---------------------------------------------------------------- 기술 조사
# 논문 원문에서만 찾는다. 웹으로 넘어가지 않는다.
SURVEY: list[Q] = [
    Q("접근_방식", "neutral",
      "{name}의 핵심 기법이 KV cache를 줄이거나 옮기는 동작 원리", None),
    Q("보고_수치", "neutral",
      "{name} 논문이 초록과 결론에 적은 압축률, 처리량 향상, 속도 향상 수치와 비교 기준선", None),
    Q("비교_기준선", "neutral",
      "{name}이 서론과 평가 절에서 다른 기준선과 비교해 적은 속도 향상 배수. "
      "speedup over baseline, times faster, improvement over", None),
    Q("성립_조건", "neutral",
      "{name}의 실험 설정과 전제. 모델 크기, 문맥 길이, 하드웨어 구성, 비교 대상", None),
    Q("저자_한계", "neutral",
      "{name} 기법 자체가 제약되는 지점. 저자가 constrained, limitation, overhead, "
      "bottleneck, future work로 언급한 대목", None),
]

# ---------------------------------------------------------------- 시장성 관점
MARKET: list[Q] = [
    Q("채택_폭", "pro",
      "{name}을 지원하는 추론 엔진이나 구현체에 대한 언급",
      "{search} implementation support in vLLM SGLang TensorRT-LLM production deployment"),
    Q("채택_깊이", "neutral",
      "{name} 구현이 기본 설정인지 선택 사항인지에 대한 언급",
      "{search} default enabled or experimental feature stability status documentation"),
    Q("진입_비용", "con",
      "{name}을 도입할 때 필요한 재학습, 커널 구현, 하드웨어 교체 같은 선행 작업",
      "{search} adoption barrier requires pretraining from scratch or new hardware purchase cost"),
    Q("생태계_지지", "pro",
      "{name}을 기준선으로 삼거나 후속으로 확장한 연구에 대한 언급",
      "{search} adopted by other models follow-up research standard specification consortium"),
    Q("채택_저조", "con",
      "{name}의 채택을 가로막는 요인에 대한 언급",
      "{search} adoption slower than expected criticism skepticism delayed rollout limitations"),
    Q("시장_수치", "neutral",
      None,
      "{market_category} market size forecast billion CAGR research firm report"),
]

# ---------------------------------------------------------------- 도메인 관점
DOMAIN: list[Q] = [
    Q("병목_일치", "neutral",
      "{name}이 장문맥 멀티턴 환경에서 겨냥하는 병목. 메모리 용량인가 대역폭인가 지연인가",
      "{search} long context multi-turn agent workload bottleneck memory capacity bandwidth"),
    Q("자원_전제", "con",
      "{name}을 쓰려면 갖춰야 하는 하드웨어와 소프트웨어 전제",
      "{search} hardware prerequisites deployment requirements infrastructure needed"),
    Q("정확도_허용치", "con",
      "{name} 적용 시 보고된 정확도 변화와 그 측정 조건",
      "{search} accuracy degradation quality loss long context reasoning benchmark"),
    Q("운영_비용", "con",
      "{name} 도입 후 늘어나는 운영 부담. 별도 관리, 추가 계층, 복잡도",
      "{search} operational complexity maintenance overhead production serving"),
    Q("도메인_적합", "pro",
      "{name}이 긴 문맥과 여러 턴에서 유리하게 측정된 결과",
      "{search} long context 100K tokens multi-turn KV cache reuse measured benefit"),
    Q("증거_성숙도", "neutral",
      "{name}의 성능 수치를 누가 측정했는가. 저자 자체 실험인가 외부 검증인가",
      "{search} third party independent reproduction benchmark verification"),
]

CONTRAST: list[Q] = [
    Q("대조_도메인", "neutral",
      None,
      "{search} applicability to {domain_en}. can it run in this environment and what blocks it"),
]

DOMAIN_EN = {
    "온디바이스 추론 (스마트폰과 엣지, 전력과 메모리 제약)":
        "on-device inference on smartphones and edge devices with power and memory constraints",
    "클라우드 대규모 동시 서빙 (수천 세션 동시 처리, 비용 민감)":
        "cloud scale concurrent serving of thousands of sessions with cost sensitivity",
}


def render(qs: list[Q], tech: dict, domain: str = "", domain_en: str = "") -> list[tuple[Q, str | None, str | None]]:
    """(질의정의, 문서풀질의, 웹질의)를 돌려준다. 해당 없으면 None."""
    out = []
    for q in qs:
        pool = q.pool.format(name=tech["search_name"]) if q.pool else None
        web = q.web.format(search=tech["search_name"],
                           market_category=tech["market_category"],
                           domain_en=domain_en) if q.web else None
        out.append((q, pool, web))
    return out
