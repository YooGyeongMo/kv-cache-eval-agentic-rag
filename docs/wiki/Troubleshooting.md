# Troubleshooting

[Home](Home.md) | [Architecture](Architecture.md) | [Evaluation](Evaluation.md) | [Tradeoffs](Tradeoffs.md) | [Schedule](Schedule.md)

레드팀을 5회 돌리며 나온 실패와 검증기 자체의 결함을 기록합니다. 대부분은 `tests/test_verify.py`에 회귀 테스트로 고정되어 있습니다.

## 파싱과 검색

### 2단 조판 논문의 열이 섞였다 ([#1](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/1))
- 현상: ITME 논문을 쪽 단위로 정렬해 읽자 `ealan architectureactive area offorexploration...`처럼 두 열이 같은 높이에서 번갈아 섞임.
- 원인: 쪽 전체를 한 번에 정렬하면 왼쪽 열과 오른쪽 열의 같은 줄이 이어 붙음.
- 해결: 가운데 띠를 가로지르는 블록 비율이 6% 미만이면 2단으로 보고, 왼쪽 열을 다 읽은 뒤 오른쪽 열로 넘어감 (`is_two_column`, `read_page`). 단어 붙음 0.033에서 0.016, 수치 70종 중 69종 보존.

### 한국어 질의에서 하이브리드가 점수를 깎았다 ([#2](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/2))
- 현상: bge-m3 한국어 질의 Hit@5가 조밀 0.85, 하이브리드 0.55.
- 원인: BM25가 한국어 어절로 영문 문서를 찾으면 잡음만 더함.
- 해결: 검색 직전 질의를 영어로 옮기는 `localize` 노드. 영어 하이브리드 Hit@5 0.95, MRR 0.930.

### 웹 검색 한도 소진을 숨기고 정상 종료했다 ([#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14))
- 현상: 시장성 관점이 통째로 "정보 없음"인 보고서.
- 원인: 감사가 찬반 개수만 세고 근거의 종류를 보지 않음.
- 해결: 시장성 웹 근거 0건이면 감사 실패. `rag/webcache.py`로 재실행 시 한도를 쓰지 않게 함.

## LLM 출력

### 참고문헌 날조 ([#5](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/5))
- 현상: `arXiv:2406.00000`, `arXiv:2606.00000` 같은 없는 번호, ITME 저자 `Kim et al.`, 무관한 IMS Global 문서.
- 해결: 참고문헌을 LLM에서 떼어 냄. 본문 꼬리표만 모아 `pool.json`과 근거 원장에서 서지를 붙이고, `tools/qa.py`가 arXiv 원본과 제목, 저자 수를 대조.

### 기법과 모델을 혼동, 정식 명칭을 지어냄 ([#6](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/6))
- 현상: MLA의 자원 전제로 "H800 8GPU, NVLink 필요"를 적음. 논문이 자기 모델을 학습할 때 쓴 장비. ITME를 `Inter-Tier Memory Expansion`으로 적음.
- 왜 나쁜가: 소프트웨어 진영인 MLA가 하드웨어 의존이 큰 기술로 읽혀 보고서의 대조 축이 뒤집힘.
- 해결: 프롬프트 공통 규칙 7, 8, 10번 추가, 보고서 생성 단계에 정식 명칭 목록 직접 전달. `Inter-Tier` 0회, `Inference Tiered` 3회로 교정. 프롬프트만으로는 다른 형태로 재발해 코드 검사로 보강 (아래).

### 꼬리표가 받치지 않는 인용 ([#7](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/7))
- 현상: `Llama2-7B 변환 시 3% 정확도 손실 보고[MLA p.6]`. 그 쪽은 어텐션 구조 설명.
- 원인: 검증기가 꼬리표의 존재만 확인.
- 해결: 문장, 표 칸 단위로 꼬리표가 가리키는 원문 안에 수치와 고유 표기가 있는지 대조.

### 단위 변환과 죽은 링크 ([#7](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/7))
- `863.7 million`을 `8억6370만 달러`로 바꿔 적어 대조가 막힘. `\d+\s*(억|만)\s*달러` 패턴 금지.
- 참고문헌 9건 중 3건이 404, 403. 링크를 지우면 본문 꼬리표와 짝이 어긋나므로 지우지 않고 "접속 확인 실패"를 명시.
- 상업 조사기관 요약을 논문과 같은 무게로 씀. 웹 근거에 일차, 시장조사, 이차 등급을 붙임.

### `64배`가 숫자만 맞아서 통과 ([#13](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/13))
- 현상: `KV cache를 64배 이상 줄여 [MLA p.6]`. 그 쪽의 `64`는 헤드 수.
- 해결: `backed_by()`가 단위 문맥까지 확인. 같은 칸에 웹 출처가 있으면 문장은 두고 잘못 붙은 논문 꼬리표만 뗌.

### 공개 연도 뒤바뀜 ([#11](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/11))
- 현상: `MLA 2026년, ITME 2024년`. 실제는 반대.
- 원인: 연도는 흔한 숫자라 수치 대조에서 제외해 둔 것이 구멍.
- 해결: 기술 이름 뒤 12자 안의 `YYYY년`을 `pool.json` 서지와 따로 대조.

### 비교 기준선 바꿔 적기 ([#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14))
- 현상: 9차 실행이 ITME의 `35.7%`(CPU 오프로딩 대비)를 재계산 대비로 적음. 숫자는 맞아서 수치 대조를 통과.
- 해결: 문단에 적힌 기준선(cpu-offload, recompute, nvme-of)이 근거에 나오는지 대조. 처음에는 수치 뒤쪽만 봤는데 `ITME p.10`이 CPU 오프로딩과 재계산을 한 문단에서 다뤄 오탐이 나서, 근거 전체에서 기준선 집합을 보도록 고침.

### 기술 선정 사유 날조 ([#13](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/13))
- 현상: "질의-정답 쌍을 기준으로 임베딩 후보를 평가하여 선정했다". 기술 선정과 임베딩 선정을 뒤섞음.
- 원인: 선정 사유와 제외 후보를 보고서 프롬프트에 넘기지 않음.
- 해결: `selection_note()`로 사람이 쓴 사유를 그대로 주입.

## 검증기 자체의 결함

거짓 양성이 많은 검사는 결국 꺼지므로, 오탐도 결함으로 다뤘습니다.

| 결함 | 원인 | 해결 | PR |
|---|---|---|---|
| 멀쩡한 웹 인용을 지어낸 것으로 판정 | 웹 제목을 55자에서 잘라 끝에 공백이 남고, 모델은 공백 없이 옮김 | 단어 경계에서 자르고, 양쪽 공백을 털고 비교 (`norm_cite`) | [#11](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/11) |
| `Gen5를`이 한 토큰으로 잡힘 | 정규식 `\w`가 한글 포함 | 아스키로 제한 | [#11](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/11) |
| `Gen5`를 근거에서 못 찾음 | 근거를 소문자로 바꾸고 토큰은 그대로 비교 | 대소문자 통일 | [#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14) |
| `35.7퍼센트`를 수치로 보지 않음 | 토큰 정규식이 `%`만 인식 | `퍼센트`, `배`, `×` 추가 | [#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14) |
| 정정기가 `15.5`를 `14.99`로 바꿈 | 스크래핑이 `USD16. 55 Billion`으로 끊은 것을 값으로 읽음 | 추측 교정 경로를 막고 소수점 주위 공백을 허용해 원래 값 복원 | [#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14) |
| 고칠 수 없는 지적이 재시도를 낭비 | 편향 방지 설명문(`BIAS_GUARDS`)에 금지어가 들어 있어 한계 장에 실리면 검증에 걸림 | 설명문에서 금지어를 빼고, 설명문 자체를 검사하는 조건 추가 | [#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14) |
| "더 낫다고 단정할 수 없다"가 걸림 | 금지어가 부정문까지 잡음 | 금지어 뒤 30자에 부정 표현이 있으면 통과 | [#14](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/14) |
| 검증기와 정정기 판정 불일치 | 판정 함수를 따로 가짐 | 같은 `backed_by()` 사용 | [#13](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/pull/13) |

## 실행 환경

### OpenAI 모델 접근 403 ([#9](https://github.com/YooGyeongMo/kv-cache-eval-agentic-rag/issues/9))
- 현상: `/v1/models`는 200인데 목록 0개, 모든 모델 403. 키 문제가 아니라 프로젝트 설정(모델 허용 목록, 결제) 문제.
- 대응: LLM 없이 할 수 있는 정정을 `tools/repair.py`로 분리해 중간 산출물을 확정. 크레딧 복구 후 `app.py final`로 처음부터 재생성.
