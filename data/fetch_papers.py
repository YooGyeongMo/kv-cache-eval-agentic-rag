"""문서 풀의 논문을 arXiv에서 내려받는다.

논문 PDF는 저장소에 넣지 않는다. arXiv 제출 라이선스는 arXiv에 배포 권한을
주는 것이지 제3자 재배포를 허락하는 것이 아니다. 대신 이 스크립트로 같은
파일을 그대로 가져온다.

  python data/fetch_papers.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = HERE / "pool.json"
UA = {"User-Agent": "Mozilla/5.0 (research script; arXiv paper fetch)"}


def main() -> None:
    spec = json.loads(POOL.read_text(encoding="utf-8"))
    ok = True
    for doc in spec["documents"]:
        if not doc.get("arxiv"):
            continue
        dest = HERE / doc["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"  이미 있음  {doc['doc_id']:<6} {dest.name}")
            continue
        url = f"https://arxiv.org/pdf/{doc['arxiv']}"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("PDF가 아닌 응답")
            dest.write_bytes(data)
            print(f"  받음       {doc['doc_id']:<6} {dest.name}  {len(data):,} bytes")
        except Exception as e:
            ok = False
            print(f"  실패       {doc['doc_id']:<6} {url}  {type(e).__name__}: {e}")
    if not ok:
        sys.exit("일부 논문을 받지 못했다. 위 URL을 직접 열어 내려받고 data/papers/에 둔다.")


if __name__ == "__main__":
    main()
