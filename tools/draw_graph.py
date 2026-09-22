"""그래프 구조를 그림으로 뽑는다. API 키 없이 돈다."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieve_loop import make_subgraph   # noqa: E402
from graph import build                           # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs"


class _Stub:
    """그림만 그리면 되므로 검색기와 웹 도구는 껍데기로 둔다."""
    chunks: list = []

    def search(self, *a, **kw):
        return []


def main() -> None:
    OUT.mkdir(exist_ok=True)
    stub = _Stub()

    main_graph = build(stub, lambda q: [], "", lambda s: "")
    sub_graph = make_subgraph(stub, lambda q: [])

    for name, g in [("architecture", main_graph), ("retrieval_subgraph", sub_graph)]:
        mmd = g.get_graph().draw_mermaid()
        (OUT / f"{name}.mmd").write_text(mmd, encoding="utf-8")
        try:
            png = g.get_graph().draw_mermaid_png()
            (OUT / f"{name}.png").write_bytes(png)
            print(f"  docs/{name}.png, docs/{name}.mmd")
        except Exception as e:
            print(f"  docs/{name}.mmd (PNG 실패: {type(e).__name__})")


if __name__ == "__main__":
    main()
