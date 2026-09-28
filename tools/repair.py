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

from agents.nodes import (CITE, attach_reference, backed_by,  # noqa: E402
                          norm_cite, selection_note, verify)

ROOT = Path(__file__).resolve().parent.parent
# \w는 한글까지 포함해서 Gen5를 처럼 조사가 붙는다. 아스키로 제한한다.
TOKEN = re.compile(r"[A-Za-z][A-Za-z.]*\d[A-Za-z0-9.\-]*|\d+(?:[.,]\d+)?\s*(?:%|퍼센트|배|×)")
POOL_TAG = re.compile(r"^(MLA|ITME)\s+p\.\d+$")

log: list[str] = []


def norm(tok: str) -> str:
    return re.sub(r"(%|퍼센트|배|×)$", "", re.sub(r"\s+", "", tok)).lower()


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


MERGED = re.compile(r"\[(MLA|ITME)\s+((?:p\.\d+\s*,\s*)+p\.\d+)\]")


def split_merged_tags(body: str) -> str:
    """한 괄호에 묶어 쓴 꼬리표를 하나씩 갈라 놓는다.

    [ITME p.11, p.2]처럼 쓰면 원장에 그런 꼬리표가 없어 지어낸 출처로
    판정된다. 실제로는 두 쪽을 함께 인용한 것이므로 갈라서 각각 단다.
    """
    def fix(m: re.Match) -> str:
        doc = m.group(1)
        pages = [p.strip() for p in m.group(2).split(",")]
        return "".join(f"[{doc} {p}]" for p in pages)

    out, n = MERGED.subn(fix, body)
    if n:
        log.append(f"묶어 쓴 꼬리표를 분리: {n}곳")
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
        if cands:
            # 여럿이면 가장 짧은 것을 고른다. 같은 출처의 제목이 말줄임표
            # 유무로 갈리는 경우라 어느 쪽을 골라도 같은 문서다.
            full = min(cands, key=len)
            if full != tag:
                log.append(f"꼬리표 복원: '{tag}' → '{full}'"
                           + (f" (후보 {len(cands)}개 중 최단)" if len(cands) > 1 else ""))
            return f"[{full}]"
        return m.group(0)
    return CITE.sub(fix, body)


# 검증기가 쓰는 것과 같은 단위로 쪼갠다. 단위가 어긋나면 교정한 꼬리표가
# 엉뚱한 문장에 붙어 검사를 통과하지 못한다.
SPLIT = re.compile(r"((?<=[.。])\s+|\n|\|)")


def repoint_and_drop(body: str, by_cite: dict[str, str]) -> str:
    """수치가 실제로 있는 쪽으로 꼬리표를 옮기고, 없으면 내린다.

    표 칸과 문단 속 문장을 구별한다. 표에서는 빈칸 대신 근거 없음이라고
    적는 편이 읽기 좋지만, 문단 한가운데에 그 말이 박히면 문장이 깨진다.
    실제로 "근거 없음 저자는 I/O 경합 시..."처럼 나왔다.
    """
    parts = SPLIT.split(body)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part)
            continue
        # 앞뒤 구분자가 세로줄이면 표 칸이다
        in_table = (i > 0 and parts[i - 1] == "|") or \
                   (i + 1 < len(parts) and parts[i + 1] == "|")
        out.append(repair_unit(part, by_cite, in_table))
    return "".join(out)


def rewrite_selection(body: str) -> str:
    """2장 기술 선정을 사람이 쓴 사유로 갈아 끼운다.

    과제는 기술 선정을 2안(사람이 직접)으로 하라고 했다. 그러므로 선정 사유는
    생성할 것이 아니라 옮겨 적을 것이다. 3차 보고서는 이 재료를 프롬프트로
    받지 못해 "질의-정답 쌍을 기준으로 임베딩 후보를 평가하여 선정했다"고
    적었다. 기술 선정과 임베딩 선정을 뒤섞은 문장이다.
    """
    m = re.search(r"(##\s*2\.\s*기술 선정\s*\n)(.*?)(?=\n##\s)", body, re.S)
    if not m:
        return body
    note = selection_note()
    lines = [l for l in note.split("\n") if l.strip()]
    body_text = [
        "기술 선정은 2안으로 했다. 에이전트가 고르는 방식은 쓰지 않고 사람이 "
        "문서 풀에서 진영별로 하나씩 직접 골랐다.",
        lines[1].replace("공통 선정 기준: ", "두 기술을 고른 공통 기준은 "),
    ]
    # 이름 뒤에 조사를 붙이면 받침에 따라 틀린다. "의 경우"로 통일한다.
    for l in lines[2:]:
        if l.startswith("같은 Doc Pool"):
            body_text.append("같은 문서 풀에서 뺀 후보와 그 이유는 다음과 같다.")
        elif l.startswith("  "):
            name, why = l.strip().split(":", 1)
            body_text.append(f"{name}의 경우 {why.strip()}")
        else:
            body_text.append(l)
    new = m.group(1) + "\n" + "\n\n".join(body_text) + "\n\n"
    log.append("2장 기술 선정을 사람이 쓴 선정 사유로 교체 "
               "(직전 원고는 임베딩 선정과 뒤섞여 있었다)")
    return body[:m.start()] + new + body[m.end():]


def fix_punctuation(body: str) -> str:
    """본문에 쓰지 않기로 한 문장부호를 말로 푼다.

    가운뎃점은 한국어에서 병렬 나열에 쓰인다. 메모리·전력 제약처럼.
    읽는 흐름을 끊고 문서마다 표기가 달라지므로 조사로 잇는다.
    표 안은 건드리지 않는다. 좁은 칸에서는 기호가 자연스럽다.
    """
    out, n = [], 0
    for line in body.split("\n"):
        if line.lstrip().startswith(("|", "#")):
            out.append(line)
            continue
        # 낱말 사이의 가운뎃점만 바꾼다. 앞뒤가 한글이나 영문일 때다.
        fixed, k = re.subn(r"(?<=[\w가-힣])·(?=[\w가-힣])", "과 ", line)
        # 받침이 없는 앞말이면 와로 적는다
        fixed = re.sub(r"([가-힣])과 (?=[\w가-힣])",
                       lambda m: m.group(1) + ("과 " if _has_final(m.group(1)) else "와 "),
                       fixed)
        n += k
        out.append(fixed)
    if n:
        log.append(f"가운뎃점을 조사로 품: {n}곳")
    return "\n".join(out)


# 상투적인 연결어를 말로 푼다. 뜻을 바꾸지 않고 바꿔 쓸 수 있는 것만 둔다.
# 문장을 지우거나 합치지는 않는다. 그건 사람이 읽고 판단할 일이다.
CLICHE = {
    "이를 통해": "그 결과",
    "뿐만 아니라": "그리고",
    "결론적으로": "정리하면",
    "요약하면": "정리하면",
    "살펴보면": "보면",
    "나아가": "여기에 더해",
    "본 연구는": "이 보고서는",
    "주목할 만한": "눈에 띄는",
}


def fix_cliche(body: str) -> str:
    """상투적인 연결어를 바꿔 쓴다. 표와 제목은 건드리지 않는다."""
    out, hits = [], []
    for line in body.split("\n"):
        if line.lstrip().startswith(("|", "#", ">")):
            out.append(line)
            continue
        for bad, good in CLICHE.items():
            if bad in line:
                hits.append(f"{bad} → {good}")
                line = line.replace(bad, good)
        out.append(line)
    if hits:
        log.append("상투어 교체: " + ", ".join(sorted(set(hits))))
    return "\n".join(out)


def _has_final(ch: str) -> bool:
    """한글 낱자에 받침이 있는지 본다. 와과를 가른다."""
    code = ord(ch)
    if not 0xAC00 <= code <= 0xD7A3:
        return True
    return (code - 0xAC00) % 28 != 0


# 165억5천만 달러처럼 억과 천만을 이어 쓰기도 한다.
BILLION = re.compile(r"(\d[\d,.]*)\s*억(?:\s*(\d)\s*천만)?\s*달러")


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
            if m.lastindex and m.group(2):
                eok += float(m.group(2)) * 0.1   # 5천만은 0.5억이다
        except (ValueError, IndexError):
            return m.group(0)
        billion = eok / 10.0
        for form in (f"{billion:g}", f"{billion:.1f}"):
            if f"{form} billion" in all_text:
                log.append(f"단위 복원: '{m.group(0)}' → '{form} billion 달러'")
                return f"{form} billion 달러"
        return m.group(0)

    return BILLION.sub(fix, body)


NUMERIC = re.compile(r"\d+(?:\.\d+)?\s*(?:%|퍼센트|배|×)")


def attach_missing(unit: str, by_cite: dict[str, str]) -> str:
    """꼬리표가 아예 없는 수치 문단에 출처를 찾아 붙인다.

    수치를 적고 근거를 안 단 문단이 나온다. 근거가 없으면 기준선이 맞는지
    확인할 방법조차 없다. 그 수치를 실제로 담은 쪽을 찾아 붙인다.
    찾지 못하면 손대지 않고 검증기가 잡게 둔다.
    """
    if CITE.search(unit) or not NUMERIC.search(unit):
        return unit
    toks = set(TOKEN.findall(unit))
    owners: list[str] = []
    for tok in toks:
        found = sorted((c for c in by_cite
                        if POOL_TAG.match(c) and backed_by(tok, by_cite[c])),
                       key=lambda c: int(c.split("p.")[1]))
        if found and found[0] not in owners:
            owners.append(found[0])
    if not owners:
        return unit
    log.append(f"빠진 출처 부착: {', '.join(owners[:3])} "
               f"({', '.join(sorted(toks)[:3])}가 실제로 나온 쪽)")
    return unit.rstrip() + "".join(f"[{o}]" for o in owners[:3])


# 본문의 주장. 정상 표기만 본다.
BILLION_CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*billion", re.I)
# 근거 쪽. 스크래핑이 소수점 뒤에서 문장을 끊어 "USD16. 55 Billion"처럼
# 깨진다. 소수점 주위의 공백을 허용해 원래 값을 되살린다.
BILLION_SRC = re.compile(r"(\d+)\s*\.\s*(\d+)\s*billion", re.I)


def fix_wrong_billions(body: str, by_cite: dict[str, str]) -> str:
    """billion 수치를 근거의 값으로 바로잡는다.

    모델이 16.55를 15.5로 옮겨 적었다. 자릿수가 비슷해 눈으로는 잘 안
    보이지만 시장 전망치가 6퍼센트 어긋난다. 근거에 나온 billion 값 중
    가장 가까운 것으로 되돌린다. 차이가 크면 손대지 않는다.
    """
    src = " ".join(by_cite.values())
    known = {float(f"{a}.{b}") for a, b in BILLION_SRC.findall(src)}
    if not known:
        return body

    def fix(m: re.Match) -> str:
        val = float(m.group(1))
        if val in known:
            return m.group(0)
        near = min(known, key=lambda k: abs(k - val))
        # 잘못 읽은 수준일 때만 고친다. 멀면 다른 지표의 값일 수 있어 둔다.
        if abs(near - val) / max(near, 1e-9) > 0.1:
            return m.group(0)
        log.append(f"billion 수치 교정: {m.group(1)} → {near:g} (근거 값)")
        return m.group(0).replace(m.group(1), f"{near:g}")

    return BILLION_CLAIM.sub(fix, body)


def repair_unit(unit: str, by_cite: dict[str, str], in_table: bool = True) -> str:
    unit = attach_missing(unit, by_cite)
    tags = [norm_cite(t) for t in CITE.findall(unit)]
    pool_tags = [t for t in tags if POOL_TAG.match(t)]
    if not pool_tags:
        return unit

    # 판정은 검증기와 같은 함수를 쓴다. 따로 두었더니 검증기가 잡는 것을
    # 정정기가 못 보는 일이 생겼다.
    backing = " ".join(by_cite.get(t, "") for t in pool_tags)
    plain = CITE.sub("", unit)
    bad = [tok for tok in set(TOKEN.findall(plain)) if not backed_by(tok, backing)]
    if not bad:
        return unit

    # 같은 문서 안에서 그 표기를 실제로 담은 쪽을 찾는다
    doc = pool_tags[0].split()[0]
    still_bad = []
    added: list[str] = []
    for tok in bad:
        owners = sorted(
            (c for c in by_cite
             if c.startswith(doc + " p.") and backed_by(tok, by_cite[c])),
            key=lambda c: int(c.split("p.")[1]))
        if owners:
            if owners[0] not in tags and owners[0] not in added:
                added.append(owners[0])
                log.append(f"출처 교정: '{tok}' → {owners[0]} (원래 {', '.join(pool_tags)})")
        else:
            still_bad.append(tok)

    if still_bad:
        web_tags = [t for t in tags if t.startswith("웹:")]
        if web_tags:
            # 같은 칸에 웹 출처가 있으면 그 주장은 웹에서 온 것이다.
            # 문장을 지우는 대신 받치지 못하는 논문 꼬리표만 뗀다.
            for t in pool_tags:
                unit = unit.replace(f"[{t}]", "")
            log.append(f"잘못 붙은 논문 꼬리표 제거: {', '.join(pool_tags)} "
                       f"({', '.join(sorted(still_bad))}는 웹 출처의 주장이다)")
            return re.sub(r"\s{2,}", " ", unit)
        log.append(f"근거 없음으로 내림: {', '.join(sorted(still_bad))} "
                   f"(꼬리표 {', '.join(pool_tags)}) — 문서 풀 어디에도 없음")
        if in_table:
            return " 근거 없음 " if unit.startswith(" ") or unit.endswith(" ") else "근거 없음"
        return ""        # 문단 속 문장이면 통째로 뺀다. 남은 문장으로 읽힌다

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
    body = fix_punctuation(body)
    body = fix_cliche(body)
    body = rewrite_selection(body)
    body = split_merged_tags(body)
    body = strip_pipes_in_tags(body)
    body = repair_tags(body, known)
    body = fix_years(body)
    body = undo_unit_conversion(body, by_cite)
    body = fix_wrong_billions(body, by_cite)
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
