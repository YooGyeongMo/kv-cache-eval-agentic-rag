"""보고서 마크다운을 PDF로 만든다.

과제가 보고서를 10쪽으로 제한한다. 변환한 뒤 쪽수를 세어 넘치면 알려 준다.
Chrome 헤드리스를 쓰는 이유는 한국어 조판이 안정적이고 표가 깨지지 않아서다.

  python tools/md2pdf.py outputs/report_seed0.md
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
/* 한국어 본문 조판. 읽는 사람이 오래 보게 되므로 행간을 넉넉히 두고
   자간을 약간 좁혀 덩어리가 또렷하게 보이도록 한다. */
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Apple SD Gothic Neo", "Noto Sans KR", -apple-system, sans-serif;
  font-size: 10.4pt;
  line-height: 1.85;
  letter-spacing: -0.15px;
  color: #1a1c22;
  margin: 0;
  word-break: keep-all;           /* 한국어 단어를 중간에서 끊지 않는다 */
  overflow-wrap: break-word;
}
h1 { font-size: 17pt; margin: 0 0 14px; letter-spacing: -0.6px; font-weight: 700; }
h2 {
  font-size: 13pt; font-weight: 700; letter-spacing: -0.4px;
  margin: 26px 0 12px; padding-bottom: 7px;
  border-bottom: 1.8px solid #1a1c22;
}
h3 { font-size: 11.2pt; font-weight: 600; margin: 18px 0 8px; color: #33373f; }
h2, h3 { break-after: avoid; }
p { margin: 0 0 12px; text-align: justify; text-justify: inter-word; }
p + p { margin-top: 2px; }

table {
  width: 100%; border-collapse: collapse; margin: 14px 0 18px;
  font-size: 9.2pt; line-height: 1.62; break-inside: avoid;
}
th, td {
  border: 1px solid #ccd0d9; padding: 8px 9px;
  vertical-align: top; text-align: left;
}
th { background: #eef0f5; font-weight: 600; letter-spacing: -0.2px; }
td:first-child, th:first-child { white-space: nowrap; width: 1%; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfd; }

code {
  font-family: "SF Mono", Menlo, monospace; font-size: 8.9pt;
  background: #f1f2f6; padding: 1px 5px; border-radius: 3px;
  letter-spacing: 0;
}
ul, ol { margin: 0 0 12px; padding-left: 20px; }
li { margin-bottom: 6px; line-height: 1.78; }
blockquote {
  margin: 14px 0; padding: 10px 14px;
  border-left: 3px solid #c8ccd6; background: #f8f9fb;
  color: #454a55; font-size: 9.6pt; line-height: 1.72;
}
blockquote p { margin: 0; }
/* 표지. 무엇을 읽는 문서이고 누가 냈는지가 첫 쪽에 분명해야 한다. */
.cover { margin: 0 0 30px; padding-bottom: 18px; border-bottom: 2.4px solid #1a1c22; }
.cover-kind {
  font-size: 8.8pt; font-weight: 600; letter-spacing: 0.2px;
  color: #6b7180; margin-bottom: 10px;
}
.cover-title {
  font-size: 18pt; font-weight: 700; letter-spacing: -0.8px;
  line-height: 1.38; color: #12141a;
}
.cover-rule { height: 1px; background: #dfe2ea; margin: 16px 0 14px; }
.cover-info { width: auto; border-collapse: collapse; font-size: 9.6pt; margin: 0; }
.cover-info th, .cover-info td {
  border: 0; padding: 3px 0; text-align: left; vertical-align: top;
  line-height: 1.6; background: none;
}
.cover-info th {
  width: 62px; color: #6b7180; font-weight: 600; letter-spacing: -0.1px;
  padding-right: 18px; white-space: nowrap;
}
.cover-info td { color: #1a1c22; font-weight: 600; white-space: nowrap; }
.cover-info tr:nth-child(even) td { background: none; }

/* 참고문헌. 번호를 달고 둘째 줄부터 들여써서 항목 경계를 또렷하게 한다. */
.ref { font-size: 9pt; line-height: 1.7; }
.ref ul.reflist { list-style: none; padding-left: 0; margin: 10px 0 0; }
.ref ul.reflist li {
  margin-bottom: 8px; padding-left: 30px; text-indent: -30px;
  word-break: keep-all;          /* 서지 정보를 단어 중간에서 끊지 않는다 */
  overflow-wrap: anywhere;       /* 긴 URL만 필요한 곳에서 접는다 */
}
.ref .refnum {
  display: inline-block; width: 26px; text-indent: 0;
  color: #6b7180; font-variant-numeric: tabular-nums;
}
.ref h2 + p, .ref > p { color: #6b7180; font-size: 8.8pt; margin-bottom: 4px; }
.ref em { font-style: italic; }
.ref strong { display: block; margin: 14px 0 2px; font-size: 9.4pt; color: #33373f; }
"""


def md_to_html(md: str) -> str:
    """보고서가 쓰는 범위만 다루는 작은 변환기. 제목, 표, 강조, 코드, 목록."""
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 표
        if line.lstrip().startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue

        # 인용구
        if line.lstrip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(buf))}</p></blockquote>")
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lv = len(m.group(1))
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>")
            i += 1
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
            continue

        if line.strip():
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and \
                    not re.match(r"^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*\||\s*>)", lines[i]):
                buf.append(lines[i])
                i += 1
            out.append(f"<p>{inline(' '.join(buf))}</p>")
            continue
        i += 1
    return "\n".join(out)


def inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    # 참고문헌의 학술지명이 *arXiv* 꼴로 들어온다. 굵게 처리한 뒤에 본다.
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", t)
    return t


def format_references(body: str) -> str:
    """참고문헌을 번호를 단 목록으로 바꾼다.

    본문과 같은 글머리표로 두면 출처인지 본문인지 구별되지 않는다. 번호를
    달고 둘째 줄부터 들여써서 한 항목이 어디서 끝나는지 눈에 들어오게 한다.
    """
    head = body.find('<h2>REFERENCE</h2>')
    if head < 0:
        return body
    before, after = body[:head], body[head:]

    n = [0]

    def number(m: re.Match) -> str:
        n[0] += 1
        return f'<li><span class="refnum">[{n[0]}]</span>{m.group(1)}</li>'

    after = re.sub(r"<li>(.*?)</li>", number, after, flags=re.S)
    after = after.replace("<ul>", '<ul class="reflist">')
    return before + after


def build_html(md: str, title: str, meta: str) -> str:
    """표지와 본문을 묶는다.

    보고서 목차가 SUMMARY로 시작하므로 문서 자체의 제목이 없다. 제출물에는
    무엇을 읽는 문서이고 누가 냈는지가 첫 쪽에 분명해야 하므로 표지를 얹는다.
    """
    body = md_to_html(md)
    body = format_references(body)
    body = body.replace("<h2>REFERENCE</h2>", '<h2>REFERENCE</h2><div class="ref">')
    if "REFERENCE" in md:
        body += "</div>"

    # 메타는 "제목 · 소속 · 이름 · 날짜" 꼴로 들어온다.
    parts = [p.strip() for p in meta.split("·") if p.strip()]
    heading = parts[0] if parts else title
    fields = parts[1:]

    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
        for k, v in fields_to_rows(fields))

    cover = (
        "<div class='cover'>"
        "<div class='cover-kind'>SKALA 생성형 AI 서비스 개발 과정 · RAG 1인 과제</div>"
        f"<div class='cover-title'>{html.escape(heading)}</div>"
        "<div class='cover-rule'></div>"
        f"<table class='cover-info'>{rows}</table>"
        "</div>")
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{cover}{body}</body></html>")


def fields_to_rows(fields: list[str]) -> list[tuple[str, str]]:
    """표지의 제출자 정보 줄을 항목과 값으로 가른다."""
    labels = {"캠퍼스": ("광주", "울산", "판교"),
              "분반": ("반",),
              "제출자": (),
              "제출일": ("-",)}
    rows: list[tuple[str, str]] = []
    for f in fields:
        if f in labels["캠퍼스"]:
            rows.append(("캠퍼스", f))
        elif f.endswith("반"):
            rows.append(("분반", f))
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", f):
            rows.append(("제출일", f))
        else:
            rows.append(("제출자", f))
    return rows


def page_count(pdf: Path) -> int:
    try:
        import pymupdf
        return pymupdf.open(pdf).page_count
    except Exception:
        return -1


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/report_seed0.md")
    md = src.read_text(encoding="utf-8")
    title = next((l.lstrip("# ").strip() for l in md.split("\n")
                  if l.startswith("# ")), "평가 보고서")
    meta = sys.argv[2] if len(sys.argv) > 2 else \
        "KV cache 최적화 기술 다관점 평가 보고서"

    out_html = src.with_suffix(".html")
    out_pdf = src.with_suffix(".pdf")
    out_html.write_text(build_html(md, title, meta), encoding="utf-8")

    subprocess.run([CHROME, "--headless=new", "--no-pdf-header-footer",
                    "--virtual-time-budget=8000",
                    f"--print-to-pdf={out_pdf}", f"file://{out_html.resolve()}"],
                   check=True, capture_output=True)
    n = page_count(out_pdf)
    print(f"{out_pdf}  {n}쪽")
    if n > 10:
        print(f"  [경고] 과제 제한 10쪽을 {n - 10}쪽 넘는다. 본문을 줄여야 한다.")


if __name__ == "__main__":
    main()
