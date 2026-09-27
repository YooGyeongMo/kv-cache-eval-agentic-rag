"""에이전트 노드.

에이전트를 다섯으로 나눈 기준은 "무엇을 근거로 삼는가"다. 기술 조사는 논문
원문만 본다. 시장 평가와 도메인 평가는 논문에 있을 수 없는 정보를 다루므로
웹을 연다. 평가 종합과 보고서 생성은 새 근거를 모으지 않고 앞 단계 결과만
다룬다. 근거의 출처가 같은 일을 한 에이전트에 묶어 두면 검증할 지점이
한 곳으로 모인다.

과제 범위에서 기술 성숙도 관점과 이해관계자 관점이 빠졌으므로 그 두
에이전트는 만들지 않는다.

REFERENCE는 LLM이 쓰지 않는다. 1차 실행에서 논문 제목과 arXiv 번호를
지어냈고 과제와 무관한 웹 문서까지 목록에 올렸다. 지금은 보고서 본문에
실제로 박힌 인용 꼬리표만 모아 근거 원장에서 서지 정보를 꺼내 붙인다.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import config
import prompts
from agents import queries as Q
from agents.retrieve_loop import collect
from llm import ask, ask_json
from rag.retriever import format_context
from state import EvalState

MAX_REPORT_TRY = 2
ROOT = Path(__file__).resolve().parent.parent
POOL_META = {d["doc_id"]: d for d in
             json.loads((ROOT / "data" / "pool.json").read_text(encoding="utf-8"))["documents"]}

CITE = re.compile(r"\[([^\[\]]{2,160}?)\]")

# 웹 근거의 출처 등급. 시장 평가에서 상업 조사기관 요약과 논문을 같은 무게로
# 쓰면 평가가 부풀려진다. 등급을 근거에 붙여 두고 보고서가 구별해 쓰게 한다.
PRIMARY_HOSTS = (
    "arxiv.org", "github.com", "openreview.net", "usenix.org", "acm.org",
    "ieee.org", "nvidia.com", "docs.vllm.ai", "vllm.ai", "huggingface.co",
    "skhynix.com", "samsung.com", "micron.com", "intel.com", "amd.com",
    "computeexpresslink.org", "marvell.com", "pytorch.org", "docs.sglang.ai",
)
MARKET_RESEARCH_HOSTS = (
    "marketintelo.com", "market.us", "mordorintelligence.com", "gminsights.com",
    "fortunebusinessinsights.com", "researchandmarkets.com", "marketsandmarkets.com",
    "grandviewresearch.com", "statista.com", "precedenceresearch.com",
    "alliedmarketresearch.com", "globenewswire.com",
)


def probe_urls(urls: list[str], timeout: int = 12) -> dict[str, bool]:
    """REFERENCE에 올릴 URL이 실제로 응답하는지 확인한다.

    검색 도구가 색인만 남고 사라진 쪽을 돌려주는 경우가 있다. 3차 점검에서
    참고문헌 9건 중 3건이 404와 403이었다. 죽은 링크를 말없이 목록에 올리면
    확인한 척이 되므로, 확인 실패를 명시한다. 링크를 지우지는 않는다.
    본문이 이미 그 꼬리표로 인용하고 있어 지우면 짝이 어긋난다.
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    ua = {"User-Agent": "Mozilla/5.0 (reference link check)"}

    def one(u: str) -> tuple[str, bool]:
        for method in ("HEAD", "GET"):
            try:
                req = urllib.request.Request(u, headers=ua, method=method)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return u, 200 <= r.status < 400
            except Exception as e:
                if getattr(e, "code", None) not in (405, None):
                    return u, False
        return u, False

    if not urls:
        return {}
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            return dict(ex.map(one, urls))
    except Exception:
        return {u: True for u in urls}      # 확인 자체가 막히면 판단하지 않는다


def source_tier(url: str) -> str:
    """일차 자료, 시장조사 요약, 기타 이차 자료로 가른다."""
    host = re.sub(r"^https?://(www\.)?", "", url or "").split("/")[0].lower()
    if any(host == h or host.endswith("." + h) for h in PRIMARY_HOSTS):
        return "일차"
    if any(host == h or host.endswith("." + h) for h in MARKET_RESEARCH_HOSTS):
        return "시장조사"
    return "이차"


# ---------------------------------------------------------------- 근거 원장
def _mk(tech: str, perspective: str, stance: str, full: str, cite: str,
        source_type: str, url: str = "", tier: str = "문서") -> dict:
    return {
        "id": f"E{abs(hash((tech, perspective, cite, full[:120]))) % 1000000:06d}",
        "tech": tech, "perspective": perspective, "stance": stance,
        "claim": " ".join(full.split())[:400],     # 프롬프트와 화면 표시용
        "full": " ".join(full.split()),            # 수치 대조용. 자르지 않는다
        "cite": cite.strip(), "source_type": source_type, "url": url, "tier": tier,
    }


def short_title(raw: str | None) -> str:
    """꼬리표에 쓸 제목. 단어 경계에서 자르고 끝 공백을 없앤다.

    1차에는 55자에서 그냥 잘랐는데 끝에 공백이 남아 원장의 꼬리표와 보고서의
    꼬리표가 한 글자 어긋났다. 검증기가 멀쩡한 인용을 지어낸 것으로 판정했다.
    """
    t = " ".join((raw or "제목 없음").split())
    if len(t) <= 55:
        return t
    cut = t[:55]
    if " " in cut:
        cut = cut[:cut.rindex(" ")]
    return cut.rstrip(" ,:;-")


def _web_context(items: list[dict]) -> str:
    """웹 결과를 근거 블록으로 만든다. 쪽수를 붙이지 않고 출처 등급을 적는다."""
    out = []
    for it in items:
        out.append(f"[웹: {short_title(it.get('title'))}] "
                   f"(출처 등급: {source_tier(it.get('url', ''))})\n"
                   f"{(it.get('content') or '').strip()}")
    return "\n\n---\n\n".join(out)


def _harvest(res: dict, tech: str, perspective: str, stance: str) -> list[dict]:
    ev = []
    for h in res.get("hits", [])[:4]:
        ev.append(_mk(tech, perspective, stance, h.chunk.text, h.chunk.cite, "pool"))
    for w in res.get("web", [])[:4]:
        ev.append(_mk(tech, perspective, stance, w.get("content") or "",
                      f"웹: {short_title(w.get('title'))}", "web", w.get("url", ""),
                      source_tier(w.get("url", ""))))
    return ev


def _gather(sub, tech: dict, qs: list, domain: str = "", domain_en: str = "",
            perspective: str = "", k: int = 5) -> tuple[list[str], list[dict]]:
    """질의 묶음을 돌려 근거 블록과 원장 항목을 만든다."""
    blocks, evidence = [], []
    for q, pool_q, web_q in Q.render(qs, tech, domain, domain_en):
        if pool_q:
            res = collect(sub, pool_q, k=k, where={"doc_id": tech["id"]},
                          allow_web=False, search_pool=True)
            if res.get("hits"):
                blocks.append(format_context(res["hits"], limit=2600))
            evidence += _harvest(res, tech["id"], perspective, q.stance)
        if web_q:
            res = collect(sub, web_q, allow_web=True, search_pool=False)
            if res.get("web"):
                blocks.append(_web_context(res["web"])[:2600])
            evidence += _harvest(res, tech["id"], perspective, q.stance)
    return blocks, evidence


# ---------------------------------------------------------------- 1. 기술 조사
def make_survey(sub):
    """논문 원문만 본다. 웹으로 넘어가지 않는다."""

    def survey(payload: dict) -> dict:
        tech = payload["tech"]
        blocks, evidence = _gather(sub, tech, Q.SURVEY, perspective="survey", k=6)
        data = ask_json(
            prompts.TECH_SURVEY_SYS,
            prompts.TECH_SURVEY_USER.format(
                tech=tech["full_name"], camp_label=tech["camp_label"],
                context="\n\n===\n\n".join(blocks)[:22000]),
        )
        brief = {
            "tech": tech["id"], "camp": tech["camp"],
            "approach": data.get("approach", ""),
            "reported_numbers": data.get("reported_numbers", []),
            "scope": data.get("scope", ""),
            "limits": data.get("limits", "근거 없음"),
            "evidence_ids": [e["id"] for e in evidence],
        }
        return {"tech_briefs": {tech["id"]: brief}, "evidence": evidence}

    return survey


# ---------------------------------------------------------------- 2, 3. 관점 평가
def make_perspective(sub, kind: str):
    """시장성과 도메인 적용이 같은 골격을 쓴다. 기준과 질의만 다르다."""
    qs, criteria, label, key = {
        "market": (Q.MARKET, config.MARKET_CRITERIA, "시장성", "market"),
        "domain": (Q.DOMAIN, config.DOMAIN_CRITERIA, "도메인 적용", "domain_fit"),
    }[kind]

    def perspective(payload: dict) -> dict:
        tech = payload["tech"]
        domain = payload.get("domain", "")
        blocks, evidence = _gather(sub, tech, qs, domain, perspective=kind)

        if kind == "domain":
            for cd in payload.get("contrast", []):
                b, e = _gather(sub, tech, Q.CONTRAST, cd,
                               Q.DOMAIN_EN.get(cd, cd), perspective="contrast")
                blocks += b
                evidence += e

        crit = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
        data = ask_json(
            prompts.PERSPECTIVE_SYS,
            prompts.PERSPECTIVE_USER.format(
                tech=tech["full_name"],
                search_note=(
                    f"주의: 이 기술의 고유명으로는 공개 자료가 거의 없어 웹 근거는 "
                    f"'{tech['market_category']}' 범주를 검색해 모았다. 범주의 근거를 "
                    f"이 기술의 근거로 쓰지 말고, 범주 수준의 관찰임을 문장에 밝힌다."
                    if tech["id"] == "ITME" else
                    "웹 근거가 이 기술을 직접 다루는지 범주를 다루는지 구별해 적는다."),
                perspective_label=label,
                domain_line=f"주 도메인: {domain}\n" if kind == "domain" else "",
                criteria=crit,
                context="\n\n===\n\n".join(blocks)[:26000]),
        )
        view = {
            "tech": tech["id"],
            "criterion_scores": data.get("criterion_scores", {}),
            "pro": data.get("pro", []), "con": data.get("con", []),
            "unknown": data.get("unknown", []),
            "evidence_ids": [e["id"] for e in evidence],
        }
        return {key: {tech["id"]: view}, "evidence": evidence}

    return perspective


# ---------------------------------------------------------------- 4. 감사
def audit(s: EvalState) -> dict:
    """확증편향 장치가 실제로 걸렸는지 센다.

    LLM에게 중립적이냐고 묻지 않는다. 물으면 그렇다고 답한다. 대신 수집된
    근거의 찬반 개수와 평가 문장의 찬반 개수를 세고, 한쪽만 나온 칸을 찾는다.
    """
    report = {"per_tech": {}, "flags": []}
    for tech in s["techs"]:
        tid = tech["id"]
        ev = [e for e in s.get("evidence", []) if e["tech"] == tid]
        row = {
            "evidence_total": len(ev),
            "evidence_pro": sum(1 for e in ev if e["stance"] == "pro"),
            "evidence_con": sum(1 for e in ev if e["stance"] == "con"),
            "evidence_pool": sum(1 for e in ev if e["source_type"] == "pool"),
            "evidence_web": sum(1 for e in ev if e["source_type"] == "web"),
            "web_일차": sum(1 for e in ev if e.get("tier") == "일차"),
            "web_이차": sum(1 for e in ev if e.get("tier") == "이차"),
            "web_시장조사": sum(1 for e in ev if e.get("tier") == "시장조사"),
        }
        for k, label in (("market", "시장성"), ("domain_fit", "도메인")):
            v = s.get(k, {}).get(tid, {})
            row[f"{k}_pro"], row[f"{k}_con"] = len(v.get("pro", [])), len(v.get("con", []))
            row[f"{k}_unknown"] = len(v.get("unknown", []))
            if row[f"{k}_pro"] == 0 or row[f"{k}_con"] == 0:
                report["flags"].append(
                    f"{tid}의 {label} 관점에 한쪽 근거만 있다. "
                    f"유리 {row[f'{k}_pro']}건, 불리 {row[f'{k}_con']}건.")
        report["per_tech"][tid] = row

    counts = [report["per_tech"][t["id"]]["evidence_total"] for t in s["techs"]]
    if counts and min(counts) and max(counts) / min(counts) > 2.0:
        report["flags"].append(
            f"두 기술의 근거 수가 2배 넘게 차이 난다: {counts}. 비교가 기울 수 있다.")

    report["passed"] = not report["flags"]
    return {"audit": report}


# ---------------------------------------------------------------- 5. 평가 종합
def synthesize(s: EvalState) -> dict:
    data = ask_json(
        prompts.CONFLICT_SYS,
        prompts.CONFLICT_USER.format(
            domain=s["domain"], contrast=" / ".join(s.get("contrast_domains", [])),
            market=_dump(s.get("market", {})), domain_eval=_dump(s.get("domain_fit", {}))),
        model=config.JUDGE)
    return {"conflicts": data.get("conflicts", []),
            "synthesis": {"domain_flips": data.get("domain_flips", [])}}


def _dump(d: dict) -> str:
    out = []
    for tid, v in d.items():
        out.append(f"### {tid}")
        for k, txt in (v.get("criterion_scores") or {}).items():
            out.append(f"- {k}: {txt}")
        for label, kk in (("유리", "pro"), ("불리", "con"), ("근거 없음", "unknown")):
            if v.get(kk):
                out.append(f"{label}: " + " | ".join(map(str, v[kk])))
    return "\n".join(out)[:12000]


def _dump_briefs(d: dict) -> str:
    out = []
    for tid, b in d.items():
        out.append(f"### {tid} ({b.get('camp')})")
        out.append(f"접근: {b.get('approach')}")
        out.append("보고 수치: " + " | ".join(b.get("reported_numbers", [])))
        out.append(f"성립 조건: {b.get('scope')}")
        out.append(f"저자가 밝힌 한계: {b.get('limits')}")
    return "\n".join(out)[:8000]


def _dump_list(items: list) -> str:
    return "\n".join(" / ".join(f"{k}: {v}" for k, v in it.items())
                     for it in items)[:6000] or "없음"


# ---------------------------------------------------------------- 6. 보고서
def make_report(setup_note: str, limits_fn, extra_numbers: str = ""):
    """limits_fn은 State를 받아 한계 장의 재료를 만든다.

    한계 장에는 감사 결과가 들어가야 하는데 감사는 보고서보다 앞 단계라서
    고정 문자열로 넘길 수 없다. 함수로 받아 생성 시점에 부른다.
    """

    def report(s: EvalState) -> dict:
        fix = ""
        if s.get("verify") and not s["verify"].get("passed", True):
            fix = ("\n\n직전 원고가 검증을 통과하지 못했다. 아래를 고쳐 다시 쓴다.\n"
                   + "\n".join(f"- {p}" for p in s["verify"]["problems"]))
        md = ask(
            prompts.REPORT_SYS,
            prompts.REPORT_USER.format(
                names="\n".join(
                    f"- {t['id']} = {t['full_name']} ({t['camp_label']})"
                    for t in s["techs"]),
                domain=s["domain"], contrast=" / ".join(s.get("contrast_domains", [])),
                briefs=_dump_briefs(s.get("tech_briefs", {})),
                market=_dump(s.get("market", {})),
                domain_eval=_dump(s.get("domain_fit", {})),
                conflicts=_dump_list(s.get("conflicts", [])),
                flips=_dump_list(s.get("synthesis", {}).get("domain_flips", [])),
                setup=setup_note, limits=limits_fn(s)) + fix,
            model=config.JUDGE, temperature=0.2)
        return {"report_md": attach_reference(md, s),
                "n_report": s.get("n_report", 0) + 1,
                "extra_numbers": extra_numbers}

    return report


def attach_reference(md: str, s: EvalState) -> str:
    """본문에 실제로 박힌 꼬리표만 모아 REFERENCE를 만든다.

    LLM이 쓴 REFERENCE는 버린다. 1차 실행에서 arXiv 번호를 2406.00000처럼
    지어냈고, 검색 중 걸린 무관한 문서까지 목록에 올렸다.
    """
    body = re.split(r"\n#{1,3}\s*REFERENCE\s*\n", md)[0].rstrip()

    tags = {t.strip() for t in CITE.findall(body)}
    url_by_title = {}
    for e in s.get("evidence", []):
        if e["source_type"] == "web" and e.get("url"):
            url_by_title.setdefault(e["cite"].strip(), e["url"])

    papers, webs = [], []
    for doc_id, meta in POOL_META.items():
        if any(re.match(rf"^{re.escape(doc_id)}\s+p\.\d+$", t) for t in tags):
            papers.append(
                f"- {meta['authors']}({meta['year']}). {meta['title']}. "
                f"*{meta['venue']}*, {meta['arxiv']}. {meta['url']}")

    cited_urls = [url_by_title[t] for t in sorted(tags)
                  if t.startswith("웹:") and t in url_by_title]
    alive = probe_urls(cited_urls)
    today = date.today().isoformat()
    for t in sorted(tags):
        if not (t.startswith("웹:") and t in url_by_title):
            continue
        u = url_by_title[t]
        mark = "" if alive.get(u, True) else f" (접속 확인 실패, {today} 기준)"
        webs.append(f"- {t[2:].strip()}. {u}{mark}")

    lines = ["", "## REFERENCE", "",
             "본문에 인용 꼬리표로 실제 사용한 자료만 적는다. 목록은 근거 원장에서 자동으로 뽑았다.", ""]
    lines += ["**논문**", ""] + (papers or ["- 없음"])
    lines += ["", "**웹 자료**", ""] + (webs or ["- 없음"])
    return body + "\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 7. 검증
def verify(s: EvalState) -> dict:
    """보고서를 코드로 검사한다.

    LLM 채점기를 믿지 않는 이유는 앞선 실험 때문이다. 충실성 채점기는 틀린
    숫자 23건 중 2건만, 근거 채점기는 21건 중 1건만 잡아냈다. 숫자 대조,
    금지어, 꼬리표 대조는 규칙이 분명하므로 코드가 더 잘한다.
    """
    md = s.get("report_md", "")
    body = re.split(r"\n#{1,3}\s*REFERENCE\s*\n", md)[0]
    problems: list[str] = []

    # 1) 우열 판정 표현
    for w in config.FORBIDDEN:
        if w in body:
            m = re.search(rf".{{0,40}}{re.escape(w)}.{{0,40}}", body)
            problems.append(f"우열을 판정하는 표현 '{w}'가 있다. 관찰 서술로 바꾼다. "
                            f"맥락: ...{m.group(0) if m else ''}...")

    # 2) 인용 꼬리표 대조. 근거 원장에 없는 꼬리표는 지어낸 출처다.
    #    양쪽 다 공백을 털고 비교한다. 예전 절단 방식이 꼬리표 끝에 공백을
    #    남겨, 보고서가 공백 없이 옮겨 적으면 멀쩡한 인용이 지어낸 것으로
    #    판정되던 일이 있었다.
    known = {e["cite"].strip() for e in s.get("evidence", [])}
    bad_tags = []
    for t in {x.strip() for x in CITE.findall(body)}:
        if t in known:
            continue
        if re.match(r"^(MLA|ITME)\s+p\.\d+$", t) or t.startswith("웹:"):
            bad_tags.append(t)          # 형식은 맞지만 원장에 없다
        elif re.search(r"p\.\d+|웹", t):
            bad_tags.append(t)
    if bad_tags:
        problems.append("근거 원장에 없는 인용 꼬리표가 있다: "
                        + ", ".join(sorted(bad_tags)[:8])
                        + ". 주어진 근거의 꼬리표를 그대로 써야 한다.")

    # 2-1) 꼬리표가 그 문장을 실제로 받치는지 본다.
    #
    # 꼬리표가 원장에 존재하기만 하면 통과하던 것이 빈틈이었다. 3차 실행에서
    # 웹에서 본 주장(Llama2-7B에서 3퍼센트 손실)에 논문 꼬리표 [MLA p.6]을
    # 붙였는데, 그 쪽은 어텐션 구조 설명이라 해당 내용이 없었다. 문장 단위로
    # 꼬리표가 가리키는 근거 안에 그 수치와 고유 표기가 있는지 확인한다.
    by_cite: dict[str, str] = {}
    for e in s.get("evidence", []):
        c = e["cite"].strip()
        by_cite[c] = by_cite.get(c, "") + " " + e.get("full", "")
    # \w는 한글까지 포함해서 Gen5를 처럼 조사가 붙어 버린다. 아스키로 제한한다.
    TOKEN = re.compile(r"[A-Za-z][A-Za-z.]*\d[A-Za-z0-9.\-]*|\d+(?:[.,]\d+)?\s*(?:%|배|×)")
    mismatched = []
    for unit in re.split(r"(?<=[.。])\s+|\n|\|", body):
        tags = [t.strip() for t in CITE.findall(unit)]
        pool_tags = [t for t in tags if re.match(r"^(MLA|ITME)\s+p\.\d+$", t)]
        if not pool_tags:
            continue
        backing = " ".join(by_cite.get(t, "") for t in pool_tags).lower()
        if not backing:
            continue
        plain = CITE.sub("", unit)
        for tok in set(TOKEN.findall(plain)):
            probe = re.sub(r"\s+", "", tok).lower().rstrip("%배×")
            if len(probe) < 3 or probe in backing.replace(" ", ""):
                continue
            mismatched.append(f"'{tok}' (꼬리표 {', '.join(pool_tags)})")
    if mismatched:
        problems.append(
            "꼬리표가 가리키는 쪽에 없는 표기를 그 꼬리표로 인용했다: "
            + ", ".join(sorted(set(mismatched))[:8])
            + ". 그 내용이 실제로 나온 근거의 꼬리표를 달거나 문장을 뺀다.")

    # 2-1b) 기술에 붙은 연도가 서지 정보와 맞는지.
    #
    # 수치 대조에서 연도를 제외해 두었더니 "MLA 2026년, ITME 2024년"처럼
    # 두 기술의 공개 연도가 뒤바뀐 채로 통과했다. 연도는 그 자체로는
    # 흔한 숫자라 일반 대조에 넣을 수 없으므로 따로 본다.
    years = {d: m.get("year") for d, m in POOL_META.items() if m.get("year")}
    for doc, year in years.items():
        for m in re.finditer(rf"{re.escape(doc)}[^.。\n|]{{0,12}}?(\d{{4}})\s*년", body):
            if m.group(1) != year:
                problems.append(
                    f"{doc}의 공개 연도를 {m.group(1)}년으로 적었다. "
                    f"서지 정보는 {year}년이다.")

    # 2-2) 단위를 바꿔 적었는지. 원문이 billion이면 억으로 고치지 않는다.
    for m in re.finditer(r"\d[\d,.]*\s*(억|만)\s*달러", body):
        problems.append(f"시장 수치의 단위를 바꿔 적었다: {m.group(0)}. "
                        f"근거의 원 표기를 그대로 옮긴다.")

    # 3) 수치 대조. 근거 원문에 없는 숫자는 만들어 낸 숫자다.
    # 검색 설정 표의 수치(임베딩 실측값)도 보고서가 인용할 수 있으므로 함께 센다.
    corpus = (" ".join(e.get("full", e.get("claim", "")) for e in s.get("evidence", []))
              + " " + s.get("extra_numbers", ""))
    corpus_nums = set(re.findall(r"\d+(?:[.,]\d+)?", corpus))
    text = re.sub(r"^#.*$", "", body, flags=re.M)
    text = re.sub(r"\[[^\]]*\]", "", text)                  # 꼬리표의 쪽수는 뺀다
    unsupported = []
    for tok in set(re.findall(r"\d+(?:[.,]\d+)?", text)):
        if tok in corpus_nums or len(tok) <= 1:
            continue
        if re.fullmatch(r"(20\d\d|1[0-9]|[1-9])", tok):
            continue                                        # 연도, 목차 번호, 한 자리
        unsupported.append(tok)
    if unsupported:
        problems.append("근거 원문에 없는 숫자가 있다: " + ", ".join(sorted(unsupported)[:12])
                        + ". 근거에 있는 값으로 바꾸거나 그 문장을 뺀다.")

    # 4) 관점별 평가 장에 근거가 달렸는가
    sec = re.search(r"##\s*4\..*?(?=\n##\s|\Z)", body, re.S)
    if sec and len(CITE.findall(sec.group(0))) < 6:
        problems.append("4장 관점별 평가에 인용 꼬리표가 너무 적다. 기준마다 근거를 단다.")

    # 5) 웹 근거에 쪽수가 붙었는지. 앞선 실험에서 실제로 생긴 오류다.
    for m in re.finditer(r"\[웹:[^\]]*?p\.\d+[^\]]*\]", body):
        problems.append(f"웹 근거에 쪽수가 붙었다: {m.group(0)}")

    # 6) 필수 장
    for head in ["# SUMMARY", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."]:
        if head not in body:
            problems.append(f"보고서에 '{head}' 장이 없다.")

    return {"verify": {"passed": not problems, "problems": problems,
                       "checked_numbers": len(corpus_nums),
                       "known_cites": len(known)}}
