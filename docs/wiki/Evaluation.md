# Evaluation

[Home](Home.md) | [Architecture](Architecture.md) | [Tradeoffs](Tradeoffs.md) | [Troubleshooting](Troubleshooting.md) | [Schedule](Schedule.md)

이 페이지의 수치는 모두 저장소의 결과 파일이나 스크립트 출력에서 옮겼습니다. 각 표 아래에 출처를 적었습니다.

## 1. 임베딩 모델 실측

질의 20개, 정답은 사람이 고른 문자열입니다. 정답을 LLM으로 만들면 채점자와 피채점자가 같은 계열이 되므로 손으로 골랐습니다.

| 모델 | 파라미터 | 색인(초) | 한국어 Hit@5 | 한국어 MRR | 한국어 하이브리드 Hit@5 | 영어 Hit@5 | 영어 MRR | 영어 하이브리드 Hit@5 | 영어 하이브리드 MRR |
|---|---|---|---|---|---|---|---|---|---|
| **bge-m3** | 568M | 24.6 | 0.85 | 0.767 | 0.55 | 0.95 | 0.867 | 0.95 | **0.930** |
| multilingual-e5-large | 560M | 24.2 | 0.75 | 0.677 | 0.50 | 0.95 | 0.919 | 0.95 | 0.880 |
| KURE-v1 | 568M | 24.3 | 0.80 | 0.692 | 0.55 | 0.95 | 0.906 | 0.95 | 0.897 |
| MiniLM-L12-v2 | 118M | 6.9 | 0.65 | 0.480 | 0.50 | 0.75 | 0.675 | 0.85 | 0.807 |

출처: `outputs/embed_bench.json` (`eval/embed_bench.py`).

읽은 것:
- 질의를 영어로 옮기면 상위 세 모델의 Hit@5가 0.95로 같아집니다. 질의 언어가 모델 선택보다 영향이 큽니다.
- 한국어 질의에 BM25를 섞으면 모든 모델에서 Hit@5가 떨어집니다. 영어 질의에서는 bge-m3 MRR이 0.867에서 0.930으로 오릅니다.
- 영어 조밀 검색만 보면 e5-large(0.919)가 bge-m3(0.867)보다 MRR이 높지만, 실제로 쓰는 영어 하이브리드에서는 bge-m3가 가장 높습니다. 그래서 "하이브리드 + 영어 질의" 조합 기준으로 골랐습니다.

## 2. PDF 로더 실측

| 로더 | 고아 숫자 비율 | 단어 붙음 비율 |
|---|---|---|
| PyMuPDF 기본 | 0.096 ~ 0.394 | 0.033 |
| PyMuPDF sort=True | 0.001 ~ 0.067 | 0.033 |
| PDFPlumber | 0.010 ~ 0.173 | 0.084 |
| **열 인식 + sort** | 0.039 | **0.016** |

출처: 이슈 [#1](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/1) 기록. 핵심 수치 70종 중 69종 보존 (사라진 둘은 그림 축 눈금).

2026-11-05에 `eval/loader_probe.py`를 다시 돌린 결과 중 문서 풀 두 편:

| 논문 | 로더 | 글자 수 | 고아 숫자 | 열 끊김 |
|---|---|---|---|---|
| 2405.04434 (MLA) | pymupdf | 109,323 | 0.270 | 0.120 |
| 2405.04434 (MLA) | pymupdf(sort) | 139,151 | 0.036 | 0.148 |
| 2405.04434 (MLA) | pdfplumber | 99,760 | 0.051 | 0.136 |
| 2606.12556 (ITME) | pymupdf | 77,216 | 0.102 | 0.132 |
| 2606.12556 (ITME) | pymupdf(sort) | 111,316 | 0.024 | 0.110 |
| 2606.12556 (ITME) | pdfplumber | 68,854 | 0.086 | 0.084 |

## 3. 근거 원장과 감사

| 항목 | MLA | ITME |
|---|---|---|
| 근거 합계 | 120 | 120 |
| 논문 / 웹 | 64 / 56 | 64 / 56 |
| 유리 / 불리 | 24 / 40 | 24 / 40 |
| 웹 출처 등급 (일차 / 이차 / 시장조사) | 7 / 49 / 0 | 14 / 42 / 0 |
| 시장성 평가 유리 / 불리 / 근거 없음 | 3 / 4 / 2 | 3 / 5 / 2 |
| 도메인 평가 유리 / 불리 / 근거 없음 | 5 / 3 / 2 | 4 / 5 / 2 |
| 감사 결과 | 통과 (flags 0) | 통과 (flags 0) |

출처: `outputs/state_seed0.json`의 `audit`. 근거 수가 대칭인 것은 우연이 아니라 대칭 질의 설계의 결과입니다. 유리와 불리가 다른 것(24 대 40)은 템플릿에서 불리 질의가 더 많기 때문입니다.

## 4. 순서 효과

```
$ python app.py compare
순서 효과 점검
  market     MLA    유리 3 대 2   불리 4 대 4   근거없음 2 대 2
  market     ITME   유리 3 대 3   불리 5 대 4   근거없음 2 대 3
  domain_fit MLA    유리 5 대 5   불리 3 대 4   근거없음 2 대 1
  domain_fit ITME   유리 4 대 5   불리 5 대 4   근거없음 2 대 2
  상충 지점 수 4 대 3
```

출처: 2026-11-05 재실행 (`outputs/state_seed0.json` 대 `state_seed1.json`). 순서를 바꿔도 찬반 항목 수 차이가 1건 이내였고, `tools/qa.py`의 기준(2건 이내)을 통과합니다. 웹 근거는 캐시 덕분에 두 실행이 같은 것을 씁니다 (정방향 새 검색 28회, 역방향 캐시 적중 28회, PR #14 기록).

## 5. 보고서 검증

| 단계 | seed 0 | seed 1 |
|---|---|---|
| 생성 직후 `verify` | 지적 2건: 원장에 없는 꼬리표 `웹: HyMCache`, 상투어 `이를 통해` | 지적 1건: 근거에 없는 숫자 `165` |
| 대조한 근거 수치 | 352개 | 352개 |
| 원장 꼬리표 | 68개 | 68개 |
| 정정 후 `tools/qa.py` 보고서 검증 | 지적 없음 | |

출처: `outputs/state_seed*.json`의 `verify`, 2026-11-05 `tools/qa.py` 출력.

## 6. 점검 결과 (2026-11-05 재실행)

| 도구 | 결과 |
|---|---|
| `pytest tests/` | 59 passed |
| `tools/qa.py` | 실패 0, 주의 0, 통과 12 (URL 15건 중 1건 응답 없음, 보고서에 명시됨) |
| `tools/rubric.py` | 충족 40, 미충족 0 |
| `tools/md2pdf.py outputs/report_seed0.md` | 8쪽 / 제한 10쪽 |

## 7. 재현 방법과 한계

API 키 없이 재현할 수 있는 범위:

```bash
uv sync --python 3.11 --dev
python data/fetch_papers.py --all
pytest tests/
python tools/md2pdf.py outputs/report_seed0.md
python tools/qa.py
python app.py compare
python eval/loader_probe.py
```

- 보고서 생성(`app.py run`)은 OpenAI와 Tavily 키가 필요합니다. 이 페이지의 생성 결과는 2026-09-28 실행분입니다.
- 임베딩 실측(`eval/embed_bench.py`)은 모델 4종을 내려받아야 해서 2026-11-05에는 다시 돌리지 않았고, 저장된 `outputs/embed_bench.json`을 옮겼습니다.
- 질의 20개는 작은 표본입니다. Hit@5 0.05 차이는 질의 1건 차이입니다.
