"""검증기가 잡은 문장을 기계적으로 정정한다.

보고서를 다시 생성하려면 LLM이 필요하지만, 검증기가 지적하는 세 가지는
규칙이 분명해서 LLM 없이 고칠 수 있다.

1. 꼬리표 절단 불일치
   웹 제목이 길어 보고서가 줄여 적으면 근거 원장의 꼬리표와 어긋난다.
   원장에서 앞부분이 일치하는 꼬리표를 찾아 온전한 표기로 되돌린다.

2. 꼬리표가 엉뚱한 곳을 가리킴
   93.3퍼센트를 [MLA p.6]으로 인용했는데 그 쪽에는 그 수치가 없다.
   같은 문서에서 그 수치를 실제로 담은 쪽을 찾아 꼬리표를 바꾼다.
   이것은 내용을 고치는 것이 아니라 출처를 바로잡는 것이다.

3. 어디에도 근거가 없는 주장
   Llama2-7B에서 3퍼센트 손실 같은 것은 문서 풀 어디에도 없다.
   다른 기법을 측정한 값을 평가 대상의 값으로 옮겨 적은 경우다.
   이럴 때는 그 칸을 '근거 없음'으로 내린다. 설계가 정한 원칙이
   근거가 없으면 근거 없음이라고 적는 것이기 때문이다.

참고문헌은 근거 원장에서 다시 뽑고 죽은 링크에 확인 실패를 표시한다.

  python tools/repair.py outputs/report_seed0.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.nodes import CITE, attach_reference, norm_cite, verify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# \w는 한글까지 포함해서 Gen5를 처럼 조사가 붙는다. 아스키로 제한한다.
TOKEN = re.compile(r"[A-Za-z][A-Za-z.]*\d[A-Za-z0-9.\-]*|\d+(?:[.,]\d+)?\s*(?:%|배|×)")
POOL_TAG = re.compile(r"^(MLA|ITME)\s+p\.\d+$")

log: list[str] = []


def norm(tok: str) -> str:
    return re.sub(r"\s+", "", tok).lower().rstrip("%배×")


def strip_pipes_in_tags(body: str) -> str:
    """꼬리표 안의 세로줄을 지운다.

    웹 제목에 "제목 | 사이트명" 형태가 흔한데, 표 안의 꼬리표에 그대로
    들어가면 마크다운이 칸 구분자로 읽어 표가 한 칸 밀린다. PDF에서
    기준 열이 세로로 찌그러지는 원인이었다.
    """
    def fix(m):
        inner = m.group(1)
        return f"[{inner.replace('|', '/')}]" if "|" in inner else m.group(0)
    n = len(CITE.findall(body))
    out = CITE.sub(fix, body)
    fixed = sum(1 for a, b in zip(CITE.findall(body), CITE.findall(out)) if a != b)
    if fixed:
        log.append(f"꼬리표 세로줄 제거: {fixed}곳 (표 칸이 밀리는 것을 막는다)")
    return out


def repair_tags(body: str, known: set[str]) -> str:
    """잘린 꼬리표를 원장의 온전한 표기로 되돌린다."""
    def fix(m: re.Match) -> str:
        tag = norm_cite(m.group(1))
        if tag in known:
            return m.group(0)
        if not tag.startswith("웹:"):
            return m.group(0)
        cands = {norm_cite(k) for k in known if k.startswith("웹:") and
                 (norm_cite(k).startswith(tag) or tag.startswith(norm_cite(k)))}
        if len(cands) == 1:
            full = cands.pop()
            if full != tag:
                log.append(f"꼬리표 복원: '{tag}' → '{full}'")
            return f"[{full}]"
        return m.group(0)
    return CITE.sub(fix, body)


# 검증기가 쓰는 것과 같은 단위로 쪼갠다. 단위가 어긋나면 교정한 꼬리표가
# 엉뚱한 문장에 붙어 검사를 통과하지 못한다.
SPLIT = re.compile(r"((?<=[.。])\s+|\n|\|)")


def repoint_and_drop(body: str, by_cite: dict[str, str]) -> str:
    """수치가 실제로 있는 쪽으로 꼬리표를 옮기고, 없으면 근거 없음으로 내린다."""
    parts = SPLIT.split(body)
    return "".join(p if i % 2 else repair_unit(p, by_cite)
                   for i, p in enumerate(parts))


BILLION = re.compile(r"(\d[\d,.]*)\s*억\s*달러")


def fix_years(body: str) -> str:
    """기술에 붙은 공개 연도를 서지 정보에 맞춘다.

    3차 보고서가 "MLA 2026년, ITME 2024년"이라고 적었는데 둘이 뒤바뀌었다.
    연도는 흔한 숫자라 일반 수치 대조에서 빠져 있어 그대로 통과했다.
    """
    meta = json.loads((ROOT / "data" / "pool.json").read_text(encoding="utf-8"))
    years = {d["doc_id"]: d["year"] for d in meta["documents"] if d.get("year")}

    def one(m: re.Match) -> str:
        doc, got = m.group(1), m.group(2)
        want = years.get(doc)
        if want and got != want:
            log.append(f"연도 교정: {doc} {got}년 → {want}년")
            return m.group(0).replace(got, want)
        return m.group(0)

    pat = "|".join(re.escape(d) for d in years)
    return re.sub(rf"({pat})([^.。\n|]{{0,12}}?)(\d{{4}})\s*년",
                  lambda m: one(re.match(rf"({pat})[^.。\n|]{{0,12}}?(\d{{4}})\s*년", m.group(0))) or m.group(0),
                  body)


def undo_unit_conversion(body: str, by_cite: dict[str, str]) -> str:
    """억 단위로 바꿔 적은 시장 수치를 근거의 원 표기로 되돌린다.

    13억 달러는 근거의 1.3 billion을 옮긴 것이다. 환산한 값이 근거에
    그대로 있는지 확인한 다음에만 바꾼다. 확인이 안 되면 건드리지 않는다.
    """
    all_text = " ".join(by_cite.values()).lower()

    def fix(m: re.Match) -> str:
        try:
            eok = float(m.group(1).replace(",", ""))
        except ValueError:
            return m.group(0)
        billion = eok / 10.0
        for form in (f"{billion:g}", f"{billion:.1f}"):
            if f"{form} billion" in all_text:
                log.append(f"단위 복원: '{m.group(0)}' → '{form} billion 달러'")
                return f"{form} billion 달러"
        return m.group(0)

    return BILLION.sub(fix, body)


def repair_unit(unit: str, by_cite: dict[str, str]) -> str:
    tags = [norm_cite(t) for t in CITE.findall(unit)]
    pool_tags = [t for t in tags if POOL_TAG.match(t)]
    if not pool_tags:
        return unit

    backing = norm(" ".join(by_cite.get(t, "") for t in pool_tags))
    plain = CITE.sub("", unit)
    bad = [tok for tok in set(TOKEN.findall(plain))
           if len(norm(tok)) >= 3 and norm(tok) not in backing]
    if not bad:
        return unit

    # 같은 문서 안에서 그 표기를 실제로 담은 쪽을 찾는다
    doc = pool_tags[0].split()[0]
    still_bad = []
    added: list[str] = []
    for tok in bad:
        owners = sorted(
            (c for c in by_cite
             if c.startswith(doc + " p.") and norm(tok) in norm(by_cite[c])),
            key=lambda c: int(c.split("p.")[1]))
        if owners:
            if owners[0] not in tags and owners[0] not in added:
                added.append(owners[0])
                log.append(f"출처 교정: '{tok}' → {owners[0]} (원래 {', '.join(pool_tags)})")
        else:
            still_bad.append(tok)

    if still_bad:
        log.append(f"근거 없음으로 내림: {', '.join(sorted(still_bad))} "
                   f"(꼬리표 {', '.join(pool_tags)}) — 문서 풀 어디에도 없음")
        return " 근거 없음 " if unit.startswith(" ") or unit.endswith(" ") else "근거 없음"

    if added:
        unit = unit.rstrip()
        # 기존 꼬리표 뒤에 올바른 출처를 덧붙인다
        unit = re.sub(r"(\[[^\]]+\])(\s*)$", r"\1" + "".join(f"[{a}]" for a in added) + r"\2", unit)
        if not unit.endswith("]"):
            unit += "".join(f"[{a}]" for a in added)
    return unit


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/report_seed0.md")
    seed = re.search(r"seed(\d+)", src.name)
    state_path = src.parent / f"state_seed{seed.group(1) if seed else 0}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    md = src.read_text(encoding="utf-8")

    before = verify({**state, "report_md": md})["verify"]
    print(f"정정 전 지적 {len(before['problems'])}건")
    for p in before["problems"]:
        print("   -", p[:150])

    known = {norm_cite(e["cite"]) for e in state.get("evidence", [])}
    by_cite: dict[str, str] = {}
    for e in state.get("evidence", []):
        c = norm_cite(e["cite"])
        by_cite[c] = by_cite.get(c, "") + " " + e.get("full", "")

    body = re.split(r"\n#{1,3}\s*REFERENCE\s*\n", md)[0].rstrip()
    body = strip_pipes_in_tags(body)
    body = repair_tags(body, known)
    body = fix_years(body)
    body = undo_unit_conversion(body, by_cite)
    body = repoint_and_drop(body, by_cite)

    note = ("\n\n> 이 보고서는 생성 후 검증 단계를 거쳤다. 인용 꼬리표가 가리키는 근거에 "
            "해당 내용이 없는 문장은 그 내용이 실제로 나온 출처로 꼬리표를 바로잡았고, "
            "문서 풀 어디에도 근거가 없는 항목은 근거 없음으로 내렸다. "
            "참고문헌의 링크는 접속을 확인했고 실패한 것은 그 사실을 적었다.\n")
    fixed = attach_reference(body + note, state)

    after = verify({**state, "report_md": fixed})["verify"]
    print(f"\n정정 내역 {len(log)}건")
    for l in log:
        print("   -", l)
    print(f"\n정정 후 지적 {len(after['problems'])}건")
    for p in after["problems"]:
        print("   -", p[:150])

    src.write_text(fixed, encoding="utf-8")
    print(f"\n[저장] {src}")


if __name__ == "__main__":
    main()
