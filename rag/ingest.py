"""문서 풀을 읽어 청크로 만든다.

로더는 eval/loader_probe.py 실측으로 고른다. 2단 조판 논문에서
PyMuPDF 기본 추출은 숫자가 문장에서 떨어져 나와 단독 줄이 되는 비율이
0.096에서 0.394였고, sort=True로 읽기 순서를 잡으면 0.001에서 0.067로 떨어졌다.
표의 수치를 근거로 인용해야 하는 과제라서 이 차이가 결과를 좌우한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 논문 절 제목. 청크가 어느 절에서 왔는지 남겨 두면 근거를 되짚기 쉽다.
SECTION = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z \-/&]{2,60}|"
    r"(?:Abstract|Introduction|Background|Motivation|Related Work|"
    r"Methodology|Method|Design|Implementation|Evaluation|Experiments?|"
    r"Results?|Discussion|Limitations?|Conclusions?|References))\s*$"
)


@dataclass
class Chunk:
    text: str
    doc_id: str       # 인용 표기에 쓰는 짧은 이름. 예: KIVI
    camp: str         # sw, hw, market 중 하나
    doc_type: str     # paper 또는 market
    page: int         # 1부터 센다
    section: str
    cite: str         # 보고서에 그대로 박히는 인용 꼬리표

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def is_two_column(page, tol: float = 0.06) -> bool:
    """가운데 띠를 가로지르는 블록이 거의 없으면 2단 조판으로 본다."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if len(blocks) < 4:
        return False
    w = page.rect.width
    lo, hi = w * 0.46, w * 0.54
    crossing = sum(1 for b in blocks if b[0] < lo and b[2] > hi)
    return crossing / len(blocks) < tol


def read_page(page) -> str:
    """2단이면 왼쪽 열을 다 읽고 오른쪽 열로 넘어간다.

    쪽 전체를 한 번에 정렬하면 두 열이 같은 높이에서 번갈아 섞여 나온다.
    ITME 논문에서 실제로 그랬다. 열을 나눠 읽으면 붙어 버린 단어가
    0.033에서 0.016으로 줄고, 수치 70종 중 69종이 그대로 남았다.
    단단 조판에서는 나누지 않는다. DeepSeek-V2 논문은 52쪽 중 1쪽만
    2단이라 사실상 건드리지 않는다.
    """
    if not is_two_column(page):
        return page.get_text(sort=True)
    r = page.rect
    mid = r.width / 2
    left = pymupdf.Rect(r.x0, r.y0, r.x0 + mid + 6, r.y1)
    right = pymupdf.Rect(r.x0 + mid - 6, r.y0, r.x1, r.y1)
    return page.get_text(sort=True, clip=left) + "\n" + page.get_text(sort=True, clip=right)


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """쪽 번호와 본문을 짝지어 돌려준다."""
    doc = pymupdf.open(path)
    out = []
    for i, page in enumerate(doc, start=1):
        text = read_page(page)
        # 하이픈 줄바꿈을 붙인다. long-\ncontext 가 long-context 가 되도록.
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        out.append((i, text))
    return out


def strip_references(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """참고문헌 이후를 버린다. 인용 목록이 검색 결과를 오염시킨다."""
    for idx, (_, text) in enumerate(pages):
        if re.search(r"^\s*(References|REFERENCES)\s*$", text, re.M):
            head = re.split(r"^\s*(?:References|REFERENCES)\s*$", text, maxsplit=1, flags=re.M)[0]
            kept = pages[:idx]
            if head.strip():
                kept = kept + [(pages[idx][0], head)]
            return kept
    return pages


def track_sections(pages: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    """쪽을 절 단위로 쪼개고 직전 절 제목을 따라 붙인다."""
    current = "Abstract"
    out = []
    for page_no, text in pages:
        buf: list[str] = []
        for line in text.split("\n"):
            m = SECTION.match(line)
            if m:
                if buf:
                    out.append((page_no, current, "\n".join(buf)))
                    buf = []
                current = " ".join(m.group(1).split())
            else:
                buf.append(line)
        if buf:
            out.append((page_no, current, "\n".join(buf)))
    return out


def chunk_pdf(path: Path, doc_id: str, camp: str, doc_type: str,
              chunk_size: int = 1100, overlap: int = 150) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pages = strip_references(read_pdf(path))
    chunks: list[Chunk] = []
    for page_no, section, body in track_sections(pages):
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) < 80:
            continue
        for piece in splitter.split_text(body):
            if len(piece.strip()) < 80:
                continue
            chunks.append(Chunk(
                text=piece.strip(),
                doc_id=doc_id,
                camp=camp,
                doc_type=doc_type,
                page=page_no,
                section=section,
                cite=f"{doc_id} p.{page_no}",
            ))
    return chunks


def chunk_markdown(path: Path, doc_id: str, camp: str,
                   chunk_size: int = 1100, overlap: int = 150) -> list[Chunk]:
    """웹에서 모은 시장 근거 문서. 쪽 번호가 없으므로 인용에 쪽을 붙이지 않는다.

    앞선 실험에서 웹 근거에 문서 쪽수가 잘못 따라붙는 것을 확인했다.
    그래서 doc_type으로 갈라 두고 page는 0으로 둔다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    text = path.read_text(encoding="utf-8")
    return [
        Chunk(text=p.strip(), doc_id=doc_id, camp=camp, doc_type="market",
              page=0, section="", cite=doc_id)
        for p in splitter.split_text(text) if len(p.strip()) >= 80
    ]


def build_pool(spec: dict) -> list[Chunk]:
    """config의 문서 풀 정의를 받아 청크 목록을 만든다."""
    chunks: list[Chunk] = []
    for item in spec["documents"]:
        path = DATA / item["path"]
        if not path.exists():
            raise FileNotFoundError(f"문서 풀에 없는 파일: {path}")
        if path.suffix.lower() == ".pdf":
            chunks += chunk_pdf(path, item["doc_id"], item["camp"], item.get("doc_type", "paper"))
        else:
            chunks += chunk_markdown(path, item["doc_id"], item["camp"])
    return chunks


def pool_page_count(spec: dict) -> int:
    """과제가 문서 풀을 200쪽으로 제한한다. 빌드할 때마다 확인한다."""
    total = 0
    for item in spec["documents"]:
        path = DATA / item["path"]
        if path.suffix.lower() == ".pdf":
            total += pymupdf.open(path).page_count
        else:
            # 웹 문서는 3000자를 한 쪽으로 환산한다.
            total += max(1, len(path.read_text(encoding="utf-8")) // 3000)
    return total


if __name__ == "__main__":
    spec = json.loads((ROOT / "data" / "pool.json").read_text(encoding="utf-8"))
    cs = build_pool(spec)
    print(f"문서 풀 {pool_page_count(spec)}쪽, 청크 {len(cs)}개")
    by_doc: dict[str, int] = {}
    for c in cs:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    for k, v in sorted(by_doc.items()):
        print(f"  {k:<16}{v:>5}")
