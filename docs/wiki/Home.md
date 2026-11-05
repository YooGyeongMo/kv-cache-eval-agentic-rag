# 자료 검색, 근거 검증 기반 AI 분석 보고서

논문과 웹에서 근거를 모으고, 보고서의 모든 문장을 근거 원장에 묶은 뒤 코드로 검증해서 내보내는 Multi-Agent + Agentic RAG 파이프라인입니다.
첫 적용 주제는 KV cache 최적화 기술 두 가지(소프트웨어 DeepSeek-V2 MLA, 하드웨어 ITME)를 시장성 관점과 도메인 적용 관점에서 비교하는 것입니다.

## 한눈에

| 항목 | 값 | 출처 |
|---|---|---|
| 문서 풀 | 논문 2편, 65쪽, 청크 211개 | `data/pool.json`, `outputs/embed_bench.json` |
| 근거 원장 | 240건 (기술별 120건, 논문 64, 웹 56, 유리 24, 불리 40) | `outputs/state_seed0.json` |
| 보고서 | 8쪽 (제한 10쪽), 웹 참고문헌 13건 + 논문 2편 | `tools/md2pdf.py` |
| 검증 | 생성 직후 지적 2건, 정정 후 0건 | `state_seed0.json` verify, `tools/qa.py` |
| 점검 | 제출 전 점검 12/12, 요구사항 대조 40/40, pytest 59개 | 2026-11-05 재실행 |
| 검색 품질 | bge-m3 + BM25 (RRF), 영어 질의 Hit@5 0.95, MRR 0.930 | `eval/embed_bench.py` |

## 페이지

| 페이지 | 내용 |
|---|---|
| [Architecture](Architecture.md) | 구성 요소, 메인 그래프, 검색 서브그래프, State 설계, 데이터 흐름, 실행 환경 |
| [Evaluation](Evaluation.md) | 임베딩과 로더 실측, 감사 수치, 순서 효과, 점검 결과, 재현 방법 |
| [Tradeoffs](Tradeoffs.md) | 결정 로그 15건 |
| [Troubleshooting](Troubleshooting.md) | 레드팀 1~5차와 검증기 결함 기록 |
| [Schedule](Schedule.md) | 실제 타임라인, 이슈와 PR로 일을 나눈 방식 |

## 저장소 바로가기

- 코드 진입점: `app.py` (실행), `graph.py` (그래프 조립), `agents/retrieve_loop.py` (Agentic RAG), `agents/nodes.py` (에이전트, 감사, 검증기)
- 이슈: [#1 ~ #9](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues?q=is%3Aissue)
- PR: [#10 ~ #14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pulls?q=is%3Apr)
