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
@page { size: A4; margin: 17mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
  font-size: 10.2pt; line-height: 1.62; color: #16181d; margin: 0;
  word-break: keep-all; overflow-wrap: anywhere;
}
h1 { font-size: 16pt; margin: 0 0 10px; letter-spacing: -0.4px; }
h2 { font-size: 12.6pt; margin: 20px 0 8px; padding-bottom: 5px;
     border-bottom: 1.6px solid #16181d; letter-spacing: -0.3px; }
h3 { font-size: 11pt; margin: 14px 0 6px; color: #2b2f3a; }
h2, h3 { break-after: avoid; }
p { margin: 0 0 8px; text-align: justify; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 14px;
        font-size: 9.1pt; break-inside: avoid; }
th, td { border: 1px solid #c8ccd6; padding: 5px 7px; vertical-align: top;
         text-align: left; }
th { background: #eef0f5; font-weight: 600; }
td:first-child, th:first-child { white-space: nowrap; width: 1%; }
table { table-layout: auto; }
tr:nth-child(even) td { background: #fafbfd; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 9pt;
       background: #f1f2f6; padding: 1px 4px; border-radius: 3px; }
ul, ol { margin: 0 0 9px; padding-left: 19px; }
li { margin-bottom: 3px; }
.meta { color: #6b7180; font-size: 8.8pt; margin-bottom: 16px;
        padding-bottom: 9px; border-bottom: 1px solid #dfe2ea; }
.ref { font-size: 8.9pt; line-height: 1.5; }
.ref p { margin-bottom: 4px; }
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
                    not re.match(r"^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*\|)", lines[i]):
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
    return t


def build_html(md: str, title: str, meta: str) -> str:
    body = md_to_html(md)
    # REFERENCE 장부터는 글자를 줄인다
    body = body.replace("<h2>REFERENCE</h2>", '<h2>REFERENCE</h2><div class="ref">')
    if "<h2>REFERENCE</h2>" in md or "REFERENCE" in md:
        body += "</div>"
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><div class='meta'>{html.escape(meta)}</div>{body}</body></html>")


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
