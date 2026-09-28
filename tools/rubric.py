"""과제 제약사항 대조표.

노션 실습 가이드와 1인 과제 공지에 적힌 제약을 하나씩 코드로 확인한다.
제출 직전에 돌려 빠진 것이 없는지 본다. tools/qa.py가 산출물의 품질을
본다면 이쪽은 과제 요구사항의 충족 여부를 본다.

  python tools/rubric.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

rows: list[tuple[str, str, bool | None, str]] = []


def check(group: str, item: str, ok: bool | None, detail: str = "") -> None:
    rows.append((group, item, ok, detail))


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main() -> None:
    readme = read(ROOT / "README.md")
    md = read(OUT / "report_seed0.md")
    body = re.split(r"\n#{1,3}\s*REFERENCE\s*\n", md)[0]
    spec = json.loads(read(ROOT / "data" / "pool.json") or "{}")
    docs = spec.get("documents", [])
    sub = config.SUBMISSION

    # ---------------------------------------------------- 1인 과제 범위 한정
    g = "범위 한정"
    check(g, "기술 선정은 2안(사람이 직접)",
          "2안" in body and "사람이" in body,
          "보고서 2장에 명시")
    check(g, "Doc Pool에서 진영별 각 1개",
          sorted(t["camp"] for t in config.TECHS) == ["hw", "sw"],
          f"SW={config.TECHS[0]['id']}, HW={config.TECHS[1]['id']}")
    check(g, "시장성 관점 포함", "시장성" in body, "4.1장")
    check(g, "도메인 적용 관점 포함", "도메인 적용" in body, "4.2장")
    check(g, "TRL 관점 제외", not hasattr(config, "TRL_CRITERIA"),
          "에이전트를 만들지 않음")
    check(g, "이해관계자 관점 제외", not hasattr(config, "STAKEHOLDER_CRITERIA"),
          "에이전트를 만들지 않음")

    # ---------------------------------------------------- 설계 요구
    g = "설계"
    check(g, "RAG 적용 에이전트 1개 이상",
          "기술 조사" in readme and "| O |" in readme,
          "기술 조사, 시장 평가, 도메인 평가 세 곳")
    total = sum(d.get("pages", 0) for d in docs)
    check(g, "문서 풀 200쪽 이내", total <= 200, f"{total}쪽")
    check(g, "오픈소스 임베딩 적용",
          "bge-m3" in readme.lower() or "BAAI" in readme,
          config.EMBEDDING)
    check(g, "임베딩 후보군과 선정 사유 명시",
          "리더보드" in readme and "Hit" in readme,
          "후보 4종 실측 비교표")
    check(g, "특정 도메인 선택", bool(config.PRIMARY_DOMAIN),
          config.PRIMARY_DOMAIN[:34])

    # ---------------------------------------------------- 보고서 형식
    g = "보고서"
    check(g, "SUMMARY가 맨 앞", md.lstrip().startswith("# SUMMARY"))
    check(g, "REFERENCE가 맨 뒤", "REFERENCE" in md.split("## 6.")[-1])
    for n, name in [(1, "분석 배경"), (2, "기술 선정"), (3, "기술 개요"),
                    (4, "관점별 평가"), (5, "시사점"), (6, "한계점")]:
        check(g, f"{n}장 {name}", f"## {n}." in md)
    summary = md.split("# SUMMARY")[-1].split("##")[0].strip()
    check(g, "SUMMARY 반 쪽 이내", len(summary) <= 1200, f"{len(summary)}자")
    pdf = OUT / f"RAG-Output_{sub['campus']}_{sub['class']}_{sub['author']}.pdf"
    if pdf.exists():
        import pymupdf
        n = pymupdf.open(pdf).page_count
        check(g, "10쪽 이내", n <= 10, f"{n}쪽")
    else:
        check(g, "10쪽 이내", None, "PDF 미생성")
    check(g, "우열 판정 표현 없음",
          not [w for w in config.FORBIDDEN if w in body],
          "금지어 검사 통과")
    check(g, "관점 간 상충 지점 명시",
          "상충" in body or "엇갈" in body or "갈린" in body, "5장 시사점")
    check(g, "논문 참고문헌 형식",
          bool(re.search(r"\(\d{4}\)\..*\*arXiv\*, \d{4}\.\d{4,5}", md)),
          "저자(연도). 제목. arXiv, 번호")
    check(g, "웹 참고문헌 형식",
          bool(re.search(r"\(.*\)\. \*.*\*\. [\w.]+, https?://", md)),
          "기관명(날짜). 제목. 사이트명, URL")

    # ---------------------------------------------------- README 요구
    g = "README"
    for s in ("# Subject", "## Overview", "## Selected Technologies", "## Features",
              "## Tech Stack", "## Agents", "## Architecture",
              "## Directory Structure", "## Usage", "## Contributors"):
        check(g, s.lstrip("# "), s in readme)
    check(g, "확증편향 방지 전략", "확증 편향" in readme or "확증편향" in readme)
    check(g, "Contributors에 PM/PL 없음",
          "PM" not in readme.split("## Contributors")[-1]
          and "PL" not in readme.split("## Contributors")[-1])
    check(g, "아키텍처 이미지", (ROOT / "docs" / "architecture.png").exists())

    # ---------------------------------------------------- 제출물
    g = "제출물"
    check(g, "파일명 규칙", pdf.exists(), pdf.name)
    check(g, "순서 효과 검증 실행",
          (OUT / "state_seed1.json").exists(), "seed 0과 1 두 방향")

    # ---------------------------------------------------- 출력
    width = max(len(i) for _, i, _, _ in rows) + 2
    cur = None
    fail = warn = 0
    print("\n" + "=" * 82)
    print("과제 제약사항 대조")
    print("=" * 82)
    for group, item, ok, detail in rows:
        if group != cur:
            print(f"\n[{group}]")
            cur = group
        mark = "OK  " if ok else ("??  " if ok is None else "XX  ")
        fail += ok is False
        warn += ok is None
        print(f"  {mark}{item:<{width}}{detail}")
    print("\n" + "=" * 82)
    print(f"  충족 {len(rows) - fail - warn}건, 미충족 {fail}건, 확인 필요 {warn}건")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
