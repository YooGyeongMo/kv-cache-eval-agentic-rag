"""보고서 검증기 회귀 테스트.

레드팀을 세 번 돌리며 나온 실패를 하나씩 고정한다. 검증기를 고치다가
예전에 잡던 것을 놓치는 일이 실제로 있었다. 고친 것이 계속 잡히는지,
멀쩡한 보고서를 틀렸다고 하지는 않는지 둘 다 본다.

API 키 없이 돈다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.nodes import norm_cite, short_title, source_tier, verify  # noqa: E402

SKELETON = """# SUMMARY
요약이다.

## 1. 분석 배경
배경이다.

## 2. 기술 선정
선정이다.

## 3. 기술 개요
개요다.

## 4. 관점별 평가
{body}

## 5. 시사점
시사점이다.

## 6. 한계점
한계다.
"""

EVIDENCE = [
    {"cite": "MLA p.1", "source_type": "pool", "tier": "문서",
     "full": "DeepSeek-V2 reduces the KV cache by 93.3% and boosts the maximum "
             "generation throughput to 5.76 times, saving 42.5% of training costs."},
    {"cite": "MLA p.6", "source_type": "pool", "tier": "문서",
     "full": "Multi-Head Latent Attention boosting inference efficiency. "
             "Conventional Transformer models usually adopt Multi-Head Attention."},
    {"cite": "ITME p.10", "source_type": "pool", "tier": "문서",
     "full": "ITME achieves up to a 35.7% throughput improvement over the "
             "CPU-offload baseline in these extended turns."},
    {"cite": "웹: CXL Memory Expansion Market Research Report", "source_type": "web",
     "tier": "시장조사", "url": "https://marketintelo.com/report/cxl",
     "full": "The CXL memory expansion market reaches 1.3 billion in 2025."},
]


def run(body: str, evidence: list | None = None) -> list[str]:
    state = {"report_md": SKELETON.format(body=body),
             "evidence": evidence if evidence is not None else EVIDENCE}
    return verify(state)["verify"]["problems"]


def has(problems: list[str], needle: str) -> bool:
    return any(needle in p for p in problems)


# ---------------------------------------------------------------- 통과해야 하는 경우
def test_올바른_인용은_통과한다():
    body = ("| 기준 | MLA | ITME |\n"
            "| 압축률 | KV cache를 93.3% 줄였다[MLA p.1] | 근거 없음 |\n"
            "| 처리량 | 최대 생성 처리량이 5.76배로 올랐다[MLA p.1] | "
            "CPU 오프로딩 대비 35.7% 향상이다[ITME p.10] |\n"
            "| 학습비용 | 42.5% 절감이다[MLA p.1] | 근거 없음 |\n"
            "| 구조 | 기존 어텐션을 바꾼다[MLA p.6] | 근거 없음 |\n"
            "| 시장 | 근거 없음 | 1.3 billion 달러다"
            "[웹: CXL Memory Expansion Market Research Report] |\n")
    assert run(body) == []


# ---------------------------------------------------------------- 레드팀 1차
def test_원장에_없는_꼬리표를_잡는다():
    body = "압축률이 높다[MLA p.99]. " * 6
    assert has(run(body), "근거 원장에 없는 인용 꼬리표")


def test_필수_장이_빠지면_잡는다():
    state = {"report_md": "# SUMMARY\n요약만 있다.\n", "evidence": EVIDENCE}
    problems = verify(state)["verify"]["problems"]
    assert has(problems, "## 1.")
    assert has(problems, "## 6.")


# ---------------------------------------------------------------- 레드팀 2차
@pytest.mark.parametrize("word", ["우수", "더 낫", "추천", "압도적", "뛰어난"])
def test_우열_판정_표현을_잡는다(word):
    body = f"MLA가 {word}다[MLA p.1]. " * 6
    assert has(run(body), "우열을 판정하는 표현")


def test_근거에_없는_숫자를_잡는다():
    body = "압축률이 77.7퍼센트다[MLA p.1]. " * 6
    assert has(run(body), "근거 원문에 없는 숫자")


# ---------------------------------------------------------------- 레드팀 3차
def test_꼬리표가_받치지_않는_인용을_잡는다():
    """웹에서 본 주장에 논문 꼬리표를 붙인 실제 사례를 고정한다.

    MLA p.6은 어텐션 구조 설명이라 Llama2-7B도 3퍼센트 손실도 없다.
    """
    body = ("| 기준 | MLA |\n"
            "| 정확도 | Llama2-7B 변환 시 3% 손실이다[MLA p.6] |\n"
            "| 압축률 | 93.3% 줄였다[MLA p.1] |\n"
            "| 처리량 | 5.76배다[MLA p.1] |\n"
            "| 비용 | 42.5% 절감이다[MLA p.1] |\n"
            "| 기타 | 35.7% 향상이다[ITME p.10] |\n")
    problems = run(body)
    assert has(problems, "꼬리표가 가리키는 쪽에 없는 표기")
    assert has(problems, "Llama2-7B")


def test_단위를_바꿔_적으면_잡는다():
    body = ("| 기준 | 값 |\n"
            "| 시장 | 2025년 13억 달러다[웹: CXL Memory Expansion Market Research Report] |\n"
            "| 압축률 | 93.3% 줄였다[MLA p.1] |\n"
            "| 처리량 | 5.76배다[MLA p.1] |\n"
            "| 비용 | 42.5% 절감이다[MLA p.1] |\n"
            "| 기타 | 35.7% 향상이다[ITME p.10] |\n")
    assert has(run(body), "단위를 바꿔 적었다")


def test_웹_근거에_쪽수를_붙이면_잡는다():
    body = "시장이 크다[웹: CXL Memory Expansion Market Research Report p.3]. " * 6
    assert has(run(body), "웹 근거에 쪽수가 붙었다")


def test_4장에_근거가_없으면_잡는다():
    body = "표도 인용도 없이 서술만 한다."
    assert has(run(body), "인용 꼬리표가 너무 적다")


# ---------------------------------------------------------------- 보조 함수
def test_꼬리표_제목은_단어_경계에서_잘린다():
    """55자에서 그냥 자르면 끝에 공백이 남아 원장과 보고서가 어긋났다."""
    t = short_title("DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference")
    assert len(t) <= 55
    assert t == t.rstrip()
    assert not t.endswith(",")


def test_짧은_제목은_그대로_둔다():
    assert short_title("MLA on GPU Cloud") == "MLA on GPU Cloud"


def test_빈_제목을_견딘다():
    assert short_title(None) == "제목 없음"
    assert short_title("") == "제목 없음"


@pytest.mark.parametrize("url,tier", [
    ("https://arxiv.org/abs/2405.04434", "일차"),
    ("https://github.com/vllm-project/vllm", "일차"),
    ("https://docs.vllm.ai/en/latest/", "일차"),
    ("https://news.skhynix.com/en/fms-2026/", "일차"),
    ("https://marketintelo.com/report/cxl", "시장조사"),
    ("https://www.statista.com/x", "시장조사"),
    ("https://www.spheron.network/blog/x", "이차"),
    ("", "이차"),
])
def test_출처_등급을_가른다(url, tier):
    assert source_tier(url) == tier


# ---------------------------------------------------------------- 레드팀 4차
def test_공개_연도가_뒤바뀌면_잡는다():
    """수치 대조에서 연도를 빼 두었더니 두 기술의 연도가 뒤바뀐 채 통과했다."""
    body = ("| 기준 | 값 |\n"
            "| 시점 | MLA 2026년, ITME 2024년 공개다[MLA p.1] |\n"
            "| 압축률 | 93.3% 줄였다[MLA p.1] |\n"
            "| 처리량 | 5.76배다[MLA p.1] |\n"
            "| 비용 | 42.5% 절감이다[MLA p.1] |\n"
            "| 기타 | 35.7% 향상이다[ITME p.10] |\n")
    problems = run(body)
    assert has(problems, "MLA의 공개 연도")
    assert has(problems, "ITME의 공개 연도")


def test_한글_조사가_붙은_영문_표기를_오탐하지_않는다():
    """\\w가 한글까지 포함해 'Gen5를'을 한 덩어리로 잡던 거짓 양성."""
    body = ("| 기준 | 값 |\n"
            "| 전송 | PCIe Gen5를 사용한다[ITME p.10] |\n"
            "| 압축률 | 93.3% 줄였다[MLA p.1] |\n"
            "| 처리량 | 5.76배다[MLA p.1] |\n"
            "| 비용 | 42.5% 절감이다[MLA p.1] |\n"
            "| 기타 | 35.7% 향상이다[ITME p.10] |\n")
    ev = EVIDENCE + [{"cite": "ITME p.10", "source_type": "pool", "tier": "문서",
                      "full": "ITME uses a PCIe Gen5 interface to provide high bandwidth."}]
    assert not has(run(body, ev), "Gen5")


def test_꼬리표_공백_차이를_오탐하지_않는다():
    """원장 꼬리표 끝에 공백이 남아도 같은 인용으로 본다."""
    ev = [{"cite": "웹: CXL Memory Expansion Market Research Report ",
           "source_type": "web", "tier": "시장조사",
           "url": "https://example.com", "full": "market reaches 1.3 billion"}]
    body = "시장이 1.3 billion 달러다[웹: CXL Memory Expansion Market Research Report]. " * 6
    assert not has(run(body, ev), "근거 원장에 없는 인용 꼬리표")


def test_꼬리표의_세로줄이_표를_깨지_않는다():
    """웹 제목의 '제목 | 사이트명'이 마크다운 표의 칸 구분자로 읽히던 문제."""
    assert "|" not in short_title("LLM Cost Optimization Market Size | CAGR of 26%")
    assert norm_cite("웹: A | B") == norm_cite("웹: A / B")


def test_참고문헌_기관명을_호스트에서_만든다():
    """과제가 웹 출처에 기관명과 사이트명을 요구하는데 검색 도구는 제목과 URL만 준다."""
    from agents.nodes import host_of, site_name
    assert site_name("https://arxiv.org/html/2602.21548v1") == "arXiv"
    assert site_name("https://marketintelo.com/report/cxl") == "Market Intelo"
    assert site_name("https://www.snia.org/x") == "SNIA"
    assert host_of("https://www.spheron.network/blog/x") == "spheron.network"
    assert site_name("") == "출처 미상"


# ---------------------------------------------------------------- 레드팀 5차
def test_단위가_다르면_근거로_보지_않는다():
    """'64배'에 [MLA p.6]을 달았는데 그 쪽에는 64가 차원 값으로만 있었다."""
    from agents.nodes import backed_by
    dims = "with 64 heads and a head dimension of 128 per layer"
    speed = "boosts the maximum generation throughput to 5.76 times"
    assert not backed_by("64배", dims), "차원 값 64를 배수 근거로 보면 안 된다"
    assert backed_by("5.76배", speed), "배수 근거는 받아들여야 한다"
    assert backed_by("93.3%", "reduces the KV cache by 93.3%")
    assert not backed_by("93.3%", "the layer index 93.3 appears here")


def test_기술_선정_사유가_사람이_쓴_것인가():
    """과제가 2안(사람이 직접)을 요구하므로 선정 사유는 생성물이 아니어야 한다."""
    import config
    from agents.nodes import selection_note
    note = selection_note()
    assert "2안" in note and "사람이" in note
    for t in config.TECHS:
        assert t["why"][:20] in note, f"{t['id']}의 선정 사유가 빠졌다"
    for name in config.NOT_SELECTED:
        assert name in note, f"뺀 후보 {name}이 빠졌다"


# ---------------------------------------------------------------- 레드팀 10차
def test_시장성에_웹_근거가_없으면_감사가_잡는다():
    """Tavily 한도가 차서 검색이 전부 빈손이었는데 감사를 통과한 적이 있다.

    그 결과 4.1 시장성 관점이 통째로 "정보 없음"인 보고서가 나왔다.
    근거의 개수만 세지 말고 종류까지 봐야 한다.
    """
    import config
    from agents.nodes import audit
    pool_only = [
        {"tech": t["id"], "perspective": "market", "stance": st,
         "source_type": "pool", "tier": "문서", "cite": f"{t['id']} p.1", "full": "x"}
        for t in config.TECHS for st in ("pro", "con")
    ]
    state = {
        "techs": config.TECHS, "evidence": pool_only,
        "market": {t["id"]: {"pro": ["a"], "con": ["b"], "unknown": []}
                   for t in config.TECHS},
        "domain_fit": {t["id"]: {"pro": ["a"], "con": ["b"], "unknown": []}
                       for t in config.TECHS},
    }
    r = audit(state)["audit"]
    assert not r["passed"]
    assert any("웹 근거가 한 건도 없다" in f for f in r["flags"])


def test_웹_근거가_있으면_감사를_통과한다():
    import config
    from agents.nodes import audit
    ev = []
    for t in config.TECHS:
        for st in ("pro", "con"):
            ev.append({"tech": t["id"], "perspective": "market", "stance": st,
                       "source_type": "pool", "tier": "문서",
                       "cite": f"{t['id']} p.1", "full": "x"})
            ev.append({"tech": t["id"], "perspective": "market", "stance": st,
                       "source_type": "web", "tier": "일차",
                       "cite": "웹: 제목", "full": "y", "url": "https://arxiv.org/x"})
    state = {
        "techs": config.TECHS, "evidence": ev,
        "market": {t["id"]: {"pro": ["a"], "con": ["b"], "unknown": []}
                   for t in config.TECHS},
        "domain_fit": {t["id"]: {"pro": ["a"], "con": ["b"], "unknown": []}
                       for t in config.TECHS},
    }
    assert audit(state)["audit"]["passed"]
