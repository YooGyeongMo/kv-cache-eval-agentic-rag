"""문서 풀의 논문을 arXiv에서 내려받는다.

논문 PDF는 저장소에 넣지 않는다. arXiv 제출 라이선스는 arXiv에 배포 권한을
주는 것이지 제3자 재배포를 허락하는 것이 아니다. 대신 이 스크립트로 같은
파일을 그대로 가져온다.

  python data/fetch_papers.py          문서 풀의 논문만 (평가 실행에 필요한 최소)
  python data/fetch_papers.py --all    로더 비교 실측에 쓴 Doc Pool 후보 6편 전부
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = HERE / "pool.json"
UA = {"User-Agent": "Mozilla/5.0 (research script; arXiv paper fetch)"}

# Doc Pool의 나머지 후보. 평가에는 쓰지 않고 eval/loader_probe.py의 로더
# 비교 실측에만 쓴다. 문서 풀에 들어가지 않으므로 200쪽 제한과 무관하다.
LOADER_PROBE_ONLY = {
    "2504.19874": "TurboQuant", "2402.02750": "KIVI",
    "2406.19707": "InfiniGen", "2511.00321": "PIM-CXL",
}


def main() -> None:
    spec = json.loads(POOL.read_text(encoding="utf-8"))
    docs = [d for d in spec["documents"] if d.get("arxiv")]
    if "--all" in sys.argv:
        docs += [{"doc_id": name, "arxiv": aid, "path": f"papers/{aid}.pdf"}
                 for aid, name in LOADER_PROBE_ONLY.items()]
    ok = True
    for doc in docs:
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
