"""웹 검색 결과 캐시.

같은 질의를 여러 번 던지는 구조다. 정방향과 역방향을 함께 돌리면 질의가
두 배가 되고, 프롬프트를 고쳐 다시 돌릴 때마다 또 처음부터 검색한다.
검색 도구의 무료 한도는 그렇게 소진됐고, 한도가 차자 시장성 관점이 통째로
비어 버린 보고서가 나왔다.

질의를 키로 결과를 저장해 두면 두 번째 실행부터는 한도를 쓰지 않는다.
근거가 실행마다 바뀌지 않으므로 순서 효과를 볼 때도 조건이 같아진다.

캐시는 저장소에 넣지 않는다. 검색 결과의 재배포가 되기 때문이다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "index" / "web_cache.json"
TTL_DAYS = 14


class WebCache:
    def __init__(self, path: Path = CACHE, ttl_days: int = TTL_DAYS):
        self.path = path
        self.ttl = ttl_days * 86400
        self.data: dict[str, dict] = {}
        self.hit = self.miss = 0
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, query: str) -> list[dict] | None:
        row = self.data.get(query)
        if not row:
            return None
        if time.time() - row.get("at", 0) > self.ttl:
            return None                     # 오래된 시장 정보는 다시 받는다
        self.hit += 1
        return row["results"]

    def put(self, query: str, results: list[dict]) -> None:
        if not results:
            return                          # 빈 결과를 캐시하면 실패가 굳는다
        self.miss += 1
        self.data[query] = {"at": time.time(), "results": results}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False),
                             encoding="utf-8")

    def summary(self) -> str:
        return f"웹 검색 캐시 적중 {self.hit}회, 새로 검색 {self.miss}회"
