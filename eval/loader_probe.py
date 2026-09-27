"""2단 조판 논문에서 로더별 읽기 순서와 숫자 보존을 비교한다."""
import re, sys, pymupdf, pdfplumber

PAPERS = {"2402.02750": "KIVI", "2405.04434": "DeepSeek-V2", "2402.02750b": None,
          "2406.19707": "InfiniGen", "2504.19874": "TurboQuant",
          "2511.00321": "PIM-CXL", "2606.12556": "ITME"}

def mupdf(path, sort):
    d = pymupdf.open(path)
    return "\n".join(p.get_text(sort=sort) for p in d)

def plumber(path):
    out = []
    with pdfplumber.open(path) as d:
        for p in d.pages:
            out.append(p.extract_text() or "")
    return "\n".join(out)

def orphan_number_rate(t):
    """숫자가 줄 전체를 혼자 차지하는 비율. 문장에서 떨어져 나온 신호."""
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    if not lines: return 1.0
    orphan = sum(1 for l in lines if re.fullmatch(r"[\d.,%×xX\s\-–]+", l))
    return orphan / len(lines)

def col_break_rate(t):
    """한 문장이 열 경계에서 끊긴 흔적. 소문자로 끝나고 다음 줄이 대문자로 시작."""
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    if len(lines) < 2: return 0.0
    bad = sum(1 for a, b in zip(lines, lines[1:])
              if re.search(r"[a-z,]$", a) and re.match(r"[A-Z]", b))
    return bad / len(lines)

if __name__ == "__main__":
    import os
    ALL = ["2402.02750", "2405.04434", "2406.19707",
           "2504.19874", "2511.00321", "2606.12556"]
    have = [a for a in ALL if os.path.exists(f"data/papers/{a}.pdf")]
    if len(have) < len(ALL):
        print("없는 논문은 건너뛴다. 전부 받으려면 "
              "python data/fetch_papers.py --all\n")
    print(f"{'paper':<14}{'loader':<18}{'chars':>9}{'고아숫자':>9}{'열끊김':>9}")
    for aid in have:
        path = f"data/papers/{aid}.pdf"
        for name, fn in [("pymupdf", lambda p: mupdf(p, False)),
                         ("pymupdf(sort)", lambda p: mupdf(p, True)),
                         ("pdfplumber", plumber)]:
            try:
                t = fn(path)
                print(f"{aid:<14}{name:<18}{len(t):>9}{orphan_number_rate(t):>9.3f}{col_break_rate(t):>9.3f}")
            except Exception as e:
                print(f"{aid:<14}{name:<18}  FAIL {type(e).__name__}")
        print()
