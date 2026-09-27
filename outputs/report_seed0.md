# SUMMARY

본 보고서는 장문맥 멀티턴 에이전트 서비스 도메인에서의 KV cache 병목을 완화하는 두 가지 대표적 접근법, DeepSeek-V2 Multi-head Latent Attention(MLA)과 ITME: Inference Tiered Memory Expansion(ITME)을 비교한다. MLA는 소프트웨어적 저차원 압축으로 KV cache를 줄이고, ITME는 하드웨어적 메모리 확장으로 대용량 캐시를 수용한다. 시장성 관점에서는 MLA가 주요 추론 스택에서 기본값으로 채택되어 도입 경로가 용이하다는 평가가 있으나, 실제 서비스 환경에서는 추가 커널 및 인프라 요구로 운영 복잡도가 높아질 수 있다. ITME는 기존 소프트웨어와의 호환성과 대용량 메모리 확장 가능성이 강조되지만, 실제 상용 CXL 하드웨어의 가용성 부족이 도입의 제약으로 지적된다. 도메인 전환 시 두 기술 모두 적용에 한계가 있으나, 그 원인은 서로 다르게 나타난다.

## 1. 분석 배경

장문맥 멀티턴 에이전트 서비스에서는 단일 세션의 문맥이 100K 토큰을 넘고, 대화가 이어지면서 KV cache가 턴마다 누적된다. 이로 인해 GPU 메모리 용량과 대역폭이 병목이 되며, 동시 사용자 수가 제한된다. 기존 Multi-Head Attention(MHA) 구조는 각 토큰마다 키와 값을 저장해야 하므로, 문맥이 길어질수록 KV cache의 크기가 기하급수적으로 증가한다. 이 문제를 해결하기 위해 소프트웨어적으로 KV cache를 압축하거나, 하드웨어적으로 메모리 계층을 확장하는 접근이 등장했다. MLA는 Transformer의 attention 메커니즘에서 키와 값을 저차원 잠재 벡터로 압축해 캐시 크기를 줄인다. 반면 ITME는 예측 가능한 대용량 데이터를 원격 CXL-hybrid 메모리로 오프로드하고, 하드웨어/소프트웨어 협조로 대역폭과 지연을 관리한다. 두 접근은 각각 소프트웨어적 구조 변경과 하드웨어적 인프라 확장을 전제로 하며, 도입 용이성, 운영 복잡도, 실제 적용 가능성에서 평가가 갈린다.

## 2. 기술 선정

본 비교는 DeepSeek-V2 Multi-head Latent Attention(MLA)과 ITME: Inference Tiered Memory Expansion(ITME)을 대상으로 한다. 두 기술 모두 장문맥 LLM 추론에서의 KV cache 병목 해소를 목표로 하며, 각각 소프트웨어와 하드웨어 계층에서 대표성을 가진다. 선정은 사람이 직접 문서 풀에서 질의-정답 쌍을 기준으로 임베딩 후보를 평가하여 이루어졌다. 선정 기준은 장문맥 멀티턴 워크로드에서의 실질적 병목 해소 접근의 다양성과, 시장성 및 도메인 적용 관점에서의 대조 가능성이었다.

## 3. 기술 개요

### MLA

MLA는 Transformer 프레임워크 내에서 키와 값을 저차원 잠재 벡터로 압축하는 Multi-head Latent Attention 방식을 사용한다. 이를 통해 기존 MHA 대비 KV cache를 93.3% 줄이고(DeepSeek 67B 기준), 최대 생성 처리량을 5.76배 높였으며, 학습 비용을 42.5% 절감했다고 보고했다(모두 DeepSeek 67B와 비교)[MLA p.6].[MLA p.1] MLA는 대규모 MoE 언어모델에서 128K 이상의 긴 문맥을 지원하도록 설계되었으며, 저차원 압축을 위한 구조 변경이 필요하다. 저자가 밝힌 한계는 근거가 없다.

### ITME

ITME는 LLM 추론에서 발생하는 대용량 KV cache와 모델 가중치를 예측 가능한 데이터로 분류해, 이를 원격 CXL-hybrid 메모리 확장 계층에 오프로드한다. 하드웨어/소프트웨어 협조 선행 로딩으로 저장소 및 네트워크 지연을 숨기고, PCIe Gen5 인터페이스를 사용해 고대역폭을 제공한다. 35턴 벤치마크(256 동시 대화, ShareGPT 데이터셋)에서 CPU-offload 대비 35.7% 처리량 개선을 보고했으며, CPU-offload는 128GB 호스트 메모리 소진 후 캐시 시스템이 붕괴되었다고 밝혔다[ITME p.11].[ITME p.10] ITME는 멀티티어 메모리 계층, CXL-hybrid 메모리, PCIe Gen5를 전제로 하며, 성능 변동(강한 경쟁 시 I/O stall) 외에 명시적 한계는 없다.

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준         | MLA                                                                                                          | ITME                                                                                                 |
|--------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 채택_폭      | vLLM, SGLang, TensorRT-LLM 등 주요 추론 스택의 커널 및 서빙 엔진 계층에 구현되어 있음[웹: LLM Inference Kernels for vLLM and SGLang (2026 Guide)][웹: Predictive Multi-Tier Memory Management for KV Cache in] | 기존 소프트웨어 스택(FlexGen, DeepSpeed Inference 등)과 호환되나, 주요 추론 스택에 직접 구현된 근거는 없음[ITME p.11][웹: Using CXL Fabric-Attached Memory to Enable Shared KV ..] |
| 채택_깊이    | vLLM, SGLang에서 MLA가 기본값으로 채택되어 있음[웹: LLM Inference Kernels for vLLM and SGLang (2026 Guide)][웹: MLA (Multi-head Latent Attention) - mistral.rs Document] | ITME의 기본값/옵션/실험 기능 여부에 대한 정보 없음[ITME p.11]                                       |
| 진입_비용    | 기존 MHA 기반 LLM을 재학습 없이도 소량 데이터로 미세조정해 MLA로 전환 가능. 구조 변경 필요[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any ] | CXL-hybrid 메모리 및 PCIe Gen5 도입 필요. 재학습 필요성 언급 없음[ITME p.11][ITME p.2]               |
| 생태계_지지  | DeepSeek V2, V3, GLM-4.7-Flash 등에서 사용. FlashInfer와 공동 개발. 표준/컨소시엄/독립 재현 근거 없음[웹: LLM Inference Kernels for vLLM and SGLang (2026 Guide)][웹: MLA (Multi-head Latent Attention) - mistral.rs Document] | 기존 소프트웨어 정책과 호환. ITME를 베이스라인으로 한 후속 연구/표준/독립 재현 근거 없음[ITME p.11] |
| 시장_수치    | LLM 비용 최적화 시장 성장률(연 26.7%)은 상위 범주 수치로, MLA 자체 수치는 없음[웹: LLM Cost Optimization Market Size / CAGR of 26%] | CXL 메모리 확장 시장(2025년 1.3 billion 달러, 2034년 11.8 billion 달러)은 상위 범주 수치로, ITME 자체 수치는 없음[웹: CXL Memory Expansion Market Research Report 2034] |

### 4.2 도메인 적용 관점

| 기준         | MLA                                                                                                          | ITME                                                                                                 |
|--------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 병목_일치    | KV cache를 64배 이상 줄여 메모리/대역폭 병목을 직접 완화[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in] | 장문맥 KV cache와 모델 가중치를 원격 계층에 오프로드해 메모리 용량 병목을 해결[ITME p.2]             |
| 자원_전제    | 128K 문맥에서 약 8GB KV cache 요구. DeepSeek 모델은 대규모 GPU 메모리와 병렬처리 전제[웹: The complete DeepSeek model guide / Guides] | PCIe Gen5 기반 CXL-hybrid 메모리 필요. 상용 하드웨어 가용성 제한[ITME p.11]                         |
| 정확도_허용치| 근거 없음 | 근거 없음                                                                                            |
| 운영_비용    | 쿼리 흡수 트릭, 압축 KV 캐시 포맷, 신규 커널 등 추가 인프라 필요. 운영 복잡도 증가[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV] | 사용자 수준 선행 로딩 API 등 신규 인프라 도입 가능성. 운영 복잡도 변화에 대한 구체적 기술 없음[ITME p.2] |
| 증거_성숙도  | 근거 없음 | SK hynix CMM, Gen5 PCIe 기반 FPGA 프로토타입 자체 평가. 제3자 검증 없음[ITME p.11][ITME p.2]           |

## 5. 시사점

관점에 따라 두 기술의 평가가 상충한다. MLA는 시장성 관점에서 주요 추론 스택에 기본값으로 채택되어 도입 경로가 용이하고, 기존 MHA 기반 LLM을 소량 데이터로 미세조정해 전환할 수 있다는 점이 강조된다. 그러나 도메인 적용 관점에서는 실제 서비스 환경에서 쿼리 흡수 트릭, 압축 KV 캐시 포맷, 신규 커널 등 추가 인프라가 필요해 운영 복잡도가 높아질 수 있고, DeepSeek 모델 자체가 대규모 GPU 메모리와 병렬처리를 전제로 하므로 단일 GPU 환경에서는 자원 요구가 여전히 높다[웹: The complete DeepSeek model guide / Guides][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV].

ITME는 시장성 관점에서 기존 소프트웨어 스택과의 호환성과 대용량 메모리 확장, 고대역폭 제공 등 기술적 가능성이 강조된다. 그러나 도메인 적용 관점에서는 상용 CXL 하드웨어의 제한적 가용성으로 인해 실제 도입이 어렵다는 점이 부각된다[ITME p.11]. 또한 ITME의 성능 수치는 저자 자체 보고에 기반하며, 실제 서비스 환경에서의 제3자 검증이나 실증적 데이터가 부족하다.

두 기술의 진입 비용은 정반대 방향을 보인다. MLA는 소프트웨어적 구조 변경과 일부 미세조정만으로 전환이 가능하지만, 실제 운영에서는 추가 커널과 인프라가 필요하다. ITME는 기존 소프트웨어와 호환되나, 하드웨어적 전제 조건(PCIe Gen5, CXL-hybrid 메모리)이 충족되어야 하므로 초기 도입 장벽이 높다.

도메인을 온디바이스 추론으로 바꾸면 MLA는 여전히 높은 자원 요구와 커널/인프라 요구로 인해 적용이 어렵고, ITME는 하드웨어 전제 조건 자체가 충족되지 않아 도입이 사실상 불가능하다[ITME p.11][웹: The complete DeepSeek model guide / Guides]. 반면, 클라우드 대규모 동시 서빙 도메인에서는 MLA가 메모리 및 대역폭 병목을 완화해 동시 사용자 수를 늘릴 수 있으나, ITME는 상용 CXL 하드웨어의 가용성 부족으로 도입이 제한될 수 있다[ITME p.11][MLA p.6].

## 6. 한계점

본 평가는 공개 정보에 기반해 이루어졌으며, 반도체 업계의 특성상 실제 채택 규모, 수율, 원가 등은 확인할 수 없다. 기술 성숙도(TRL)와 이해관계자 관점은 범위에서 제외했다. 확증편향을 막기 위해 두 기술에 대칭적 질문 템플릿을 적용하고, 유불리 근거를 각각 따로 수집했으나, 근거의 공개 시점 차이(MLA 2024년, ITME 2026년)로 인해 채택 현황 비교에서 시간 효과가 완전히 상쇄되지는 않는다. 웹 근거 중 일부는 개별 기술이 아닌 LLM 비용 최적화, CXL 메모리 확장 등 상위 범주에 대한 것이었으며, 해당 수치는 기술 자체의 시장성 수치로 직접 활용할 수 없다. 모든 평가 문장에는 근거 꼬리표를 부착했으나, 근거가 없는 항목은 "근거 없음"으로 명시했다. LLM 채점기와 코드 대조를 통해 수치 오류와 우열 표현을 점검했으나, 비공개 정보나 실제 서비스 환경에서의 미세한 운영 비용 변화 등은 반영하지 못했다.

> 이 보고서는 생성 후 검증 단계를 거쳤다. 인용 꼬리표가 가리키는 근거에 해당 내용이 없는 문장은 그 내용이 실제로 나온 출처로 꼬리표를 바로잡았고, 문서 풀 어디에도 근거가 없는 항목은 근거 없음으로 내렸다. 참고문헌의 링크는 접속을 확인했고 실패한 것은 그 사실을 적었다.

## REFERENCE

본문에 인용 꼬리표로 실제 사용한 자료만 적는다. 목록은 근거 원장에서 자동으로 뽑았다.

**논문**

- DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434. https://arxiv.org/abs/2405.04434
- Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556. https://arxiv.org/abs/2606.12556

**웹 자료**

- Market Intelo(작성일 미상). *CXL Memory Expansion Market Research Report 2034*. marketintelo.com, https://marketintelo.com/report/cxl-memory-expansion-market (접속 확인 실패, 2026-09-27 기준)
- arXiv(작성일 미상). *DualPath: Breaking the Storage Bandwidth Bottleneck in*. arxiv.org, https://arxiv.org/html/2602.21548v1
- arXiv(작성일 미상). *Enabling DeepSeek's Multi-Head Latent Attention in Any*. arxiv.org, https://arxiv.org/html/2502.14837v1
- Market.us(작성일 미상). *LLM Cost Optimization Market Size / CAGR of 26%*. market.us, https://market.us/report/llm-cost-optimization-market
- Spheron Network(작성일 미상). *LLM Inference Kernels for vLLM and SGLang (2026 Guide)*. spheron.network, https://www.spheron.network/blog/deploy-flashinfer-gpu-cloud-llm-inference-kernels
- Ericlbuehler(작성일 미상). *MLA (Multi-head Latent Attention) - mistral.rs Document*. ericlbuehler.github.io, https://ericlbuehler.github.io/mistral.rs/MLA.html (접속 확인 실패, 2026-09-27 기준)
- Spheron Network(작성일 미상). *Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV*. spheron.network, https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud
- arXiv(작성일 미상). *Predictive Multi-Tier Memory Management for KV Cache in*. arxiv.org, https://arxiv.org/html/2604.26968v2
- Baseten(작성일 미상). *The complete DeepSeek model guide / Guides*. baseten.co, https://www.baseten.co/resources/guide/the-complete-deepseek-model-guide
- SNIA(작성일 미상). *Using CXL Fabric-Attached Memory to Enable Shared KV ..*. snia.org, https://www.snia.org/sniadeveloper/session/19665 (접속 확인 실패, 2026-09-27 기준)
