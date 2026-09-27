"""제출 전 최종 점검.

레드팀을 돌리며 찾은 실패 방식을 하나씩 자동 검사로 바꿔 모아 둔 것이다.
사람이 매번 같은 곳을 다시 볼 필요가 없게 한다.

  python tools/qa.py                     seed 0 보고서를 점검
  python tools/qa.py --no-net            URL 접속 검사를 건너뛴다
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                    # noqa: E402
from agents.nodes import verify                  # noqa: E402
from rag.ingest import pool_page_count           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PAGE_LIMIT_POOL = 200
PAGE_LIMIT_REPORT = 10
UA = {"User-Agent": "Mozilla/5.0 (link check)"}

PASS, FAIL, WARN = "통과", "실패", "주의"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    results.append((name, PASS if ok else (WARN if ok is None else FAIL), detail))


# ---------------------------------------------------------------- 개별 점검
def check_pool() -> None:
    spec = json.loads((ROOT / "data" / "pool.json").read_text(encoding="utf-8"))
    n = pool_page_count(spec)
    check("문서 풀 쪽수", n <= PAGE_LIMIT_POOL, f"{n}쪽 / 제한 {PAGE_LIMIT_POOL}쪽")


def check_report(seed: int) -> dict:
    sp = OUT / f"state_seed{seed}.json"
    if not sp.exists():
        check(f"seed{seed} 실행 결과", False, "state 파일이 없다. app.py run 을 먼저 돌린다.")
        return {}
    state = json.loads(sp.read_text(encoding="utf-8"))
    md = (OUT / f"report_seed{seed}.md").read_text(encoding="utf-8")
    state["report_md"] = md

    v = verify(state)["verify"]
    check(f"seed{seed} 보고서 검증", v["passed"],
          "지적 없음" if v["passed"] else " / ".join(v["problems"])[:400])

    audit = state.get("audit", {})
    check(f"seed{seed} 확증편향 감사", audit.get("passed"),
          "한쪽으로 기운 항목 없음" if audit.get("passed")
          else " / ".join(audit.get("flags", []))[:300])

    body = re.split(r"\n#{1,3}\s*REFERENCE\s*\n", md)[0]
    hits = [w for w in config.FORBIDDEN if w in body]
    check(f"seed{seed} 우열 표현", not hits, "없음" if not hits else ", ".join(hits))

    for head in ["# SUMMARY", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "REFERENCE"]:
        if head not in md:
            check(f"seed{seed} 장 구성", False, f"'{head}' 없음")
            break
    else:
        check(f"seed{seed} 장 구성", True, "SUMMARY부터 REFERENCE까지 모두 있음")

    # 근거 없음 표기가 남아 있는지. 모두 채워졌다면 오히려 의심스럽다.
    n_unknown = body.count("근거 없음")
    check(f"seed{seed} 근거 없음 표기", None if n_unknown == 0 else True,
          f"{n_unknown}곳. 0이면 못 찾은 항목을 채워 넣었는지 사람이 확인한다.")
    return state


def check_links(seed: int, offline: bool) -> None:
    md = (OUT / f"report_seed{seed}.md").read_text(encoding="utf-8")
    urls = sorted(set(re.findall(r"https?://[^\s)\]]+", md)))
    check(f"seed{seed} REFERENCE 항목 수", len(urls) > 0, f"URL {len(urls)}건")
    if offline or not urls:
        return

    def probe(u: str) -> tuple[str, int | str]:
        for method in ("HEAD", "GET"):
            try:
                req = urllib.request.Request(u, headers=UA, method=method)
                with urllib.request.urlopen(req, timeout=20) as r:
                    return u, r.status
            except Exception as e:
                code = getattr(e, "code", None)
                if code and code != 405:
                    return u, code
        return u, "접속 실패"

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(probe, urls))
    dead = [(u, s) for u, s in out if not (isinstance(s, int) and 200 <= s < 400)]

    # 죽은 링크 자체는 막을 수 없다. 검색 도구가 색인만 남은 쪽을 돌려주기도 하고
    # 자료가 나중에 내려가기도 한다. 문제는 죽은 것을 숨기고 올리는 쪽이다.
    # 보고서가 접속 실패를 적어 두었으면 처리된 것으로 본다.
    undisclosed = [(u, s) for u, s in dead
                   if not re.search(rf"{re.escape(u)}[^\n]*접속 확인 실패", md)]
    check(f"seed{seed} URL 접속", not undisclosed,
          ("전부 응답" if not dead else
           f"{len(dead)}건 응답 없음. 모두 보고서에 접속 확인 실패로 명시됨")
          if not undisclosed else
          "명시 없이 죽은 링크: " + " / ".join(f"{s} {u[:60]}" for u, s in undisclosed[:4]))


def check_arxiv(offline: bool) -> None:
    """참고문헌의 arXiv 번호가 실재하고 제목과 저자가 맞는지 본다.

    1차 실행에서 LLM이 arXiv:2406.00000 같은 없는 번호를 참고문헌에 올렸다.
    지금은 서지 정보를 pool.json에 사람이 적어 두고 참고문헌을 거기서 뽑지만,
    그 적어 둔 값 자체가 틀렸을 수 있으므로 원본과 대조한다.

    export.arxiv.org API는 이 환경에서 406을 돌려주므로 초록 페이지의
    citation 메타 태그를 읽는다.
    """
    spec = json.loads((ROOT / "data" / "pool.json").read_text(encoding="utf-8"))
    if offline:
        check("arXiv 서지 대조", None, "생략 (--no-net)")
        return
    head = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml"}
    for d in spec["documents"]:
        if not d.get("arxiv"):
            continue
        try:
            url = f"https://arxiv.org/abs/{d['arxiv']}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=head), timeout=25) as r:
                html = r.read().decode("utf-8", "replace")
            m = re.search(r'name="citation_title"\s+content="([^"]+)"', html)
            authors = re.findall(r'name="citation_author"\s+content="([^"]+)"', html)
            got = " ".join((m.group(1) if m else "").split())
            want = " ".join(d["title"].split())
            ok = bool(got) and got.lower()[:55] == want.lower()[:55]
            check(f"arXiv {d['arxiv']} 서지", ok,
                  f"{d['doc_id']}: {got[:70] or '제목 조회 실패'} / 저자 {len(authors)}명")
        except Exception as e:
            check(f"arXiv {d['arxiv']} 서지", None, f"조회 실패 {type(e).__name__}")


def check_pdf(seed: int) -> None:
    pdf = OUT / f"report_seed{seed}.pdf"
    if not pdf.exists():
        check(f"seed{seed} PDF", None, "아직 변환 전. tools/md2pdf.py 를 돌린다.")
        return
    import pymupdf
    n = pymupdf.open(pdf).page_count
    check(f"seed{seed} PDF 쪽수", n <= PAGE_LIMIT_REPORT, f"{n}쪽 / 제한 {PAGE_LIMIT_REPORT}쪽")


def check_order_effect() -> None:
    a, b = OUT / "state_seed0.json", OUT / "state_seed1.json"
    if not b.exists():
        check("순서 효과 검증", None, "seed 1 실행 결과가 없다. app.py run --seed 1 을 돌린다.")
        return
    sa, sb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    diffs = []
    for key in ("market", "domain_fit"):
        for tid in sa.get(key, {}):
            va, vb = sa[key].get(tid, {}), sb.get(key, {}).get(tid, {})
            for field in ("pro", "con", "unknown"):
                na, nb = len(va.get(field, [])), len(vb.get(field, []))
                if abs(na - nb) > 2:
                    diffs.append(f"{key}.{tid}.{field} {na}대{nb}")
    check("순서 효과 검증", not diffs,
          "순서를 바꿔도 찬반 항목 수가 2건 이내로 유지됨" if not diffs
          else "순서에 따라 흔들림: " + ", ".join(diffs[:6]))


# ---------------------------------------------------------------- 실행
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-net", action="store_true")
    a = ap.parse_args()

    check_pool()
    check_report(a.seed)
    check_links(a.seed, a.no_net)
    check_arxiv(a.no_net)
    check_pdf(a.seed)
    check_order_effect()

    width = max(len(n) for n, _, _ in results) + 2
    print("\n" + "=" * 78)
    print("제출 전 최종 점검")
    print("=" * 78)
    for name, verdict, detail in results:
        print(f"  [{verdict}] {name:<{width}} {detail}")
    n_fail = sum(1 for _, v, _ in results if v == FAIL)
    n_warn = sum(1 for _, v, _ in results if v == WARN)
    print("=" * 78)
    print(f"  실패 {n_fail}건, 주의 {n_warn}건, 통과 {len(results) - n_fail - n_warn}건")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
