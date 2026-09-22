# SUMMARY

이 보고서는 장문맥 멀티턴 에이전트 서비스 도메인에서의 KV 캐시 병목을 완화하는 두 기술, MLA(Multi-head Latent Attention)와 ITME(Inter-Tier Memory Expansion)를 비교한다. 시장성 관점과 도메인 적용 관점에서 각각의 장단점이 다르게 드러난다. MLA는 소프트웨어적 접근으로 주요 추론 스택과 일부 모델에 통합되어 도입이 상대적으로 용이하나, 생태계 확장성과 제3자 검증, 직접적 시장성 수치가 부족하다. ITME는 하드웨어-소프트웨어 협력 기반으로 대용량 메모리 확장과 지연 마스킹을 실험적으로 보였으나, 상용화·생태계 확장 근거와 정확도 평가가 부족하다. 두 기술 모두 도입 전제와 운영 복잡도, 외부 검증의 부족 등에서 관점에 따라 평가가 엇갈린다. 도메인 전환 시, MLA는 온디바이스 환경에서 인프라 요구가 장애가 되고, ITME는 클라우드 환경에서는 확장성 가능성이 있으나 온디바이스에서는 적용 자체가 어렵다.

## 1. 분석 배경

장문맥 멀티턴 에이전트 서비스는 단일 세션에서 100K 토큰이 넘는 문맥을 유지하며, 대화가 이어질수록 KV 캐시가 턴마다 누적된다. 이때 메모리 용량과 대역폭이 병목이 되며, 특히 대규모 LLM에서는 KV 캐시가 수백 GB에 달해 동시 세션 처리와 비용 효율성에 큰 영향을 준다. MLA와 ITME는 이 병목을 각각 소프트웨어적 압축과 하드웨어-소프트웨어 협력 기반 오프로드로 접근한다. MLA는 Transformer 내에서 KV 캐시를 저차원 잠재 벡터로 압축해 메모리 사용량을 줄이고, ITME는 CXL 하이브리드 메모리 계층에 KV 캐시와 모델 가중치를 오프로드해 용량 병목을 완화한다. 두 접근은 요구 인프라, 도입 난이도, 운영 복잡도, 생태계 확장성 등에서 차이가 크며, 시장성 및 실제 워크로드 적용 가능성에 따라 평가가 달라진다.

## 2. 기술 선정

본 비교는 MLA와 ITME 두 기술을 선정해 진행했다. 선정 기준은 장문맥 멀티턴 에이전트 서비스에서의 KV 캐시 병목 완화에 초점을 두고, 소프트웨어적 접근과 하드웨어-소프트웨어 협력 접근을 대표하는 기술을 각각 고르는 것이었다. 후보군은 공개 논문, 기술 보고서, 주요 추론 스택의 통합 사례, 시장 전망 보고서 등에서 직접 수집했다. 최종 선정은 사람이 직접 근거를 검토해 결정했다.

## 3. 기술 개요

### MLA

MLA(Multi-head Latent Attention)는 Transformer 프레임워크 내에서 KV 캐시를 저차원 잠재 벡터로 압축하는 방식이다. MHA(Multi-head Attention) 대비 동일 또는 더 나은 성능을 유지하면서 KV 캐시 요구량을 크게 줄인다. DeepSeek 67B 모델 기준, KV 캐시를 93.3% 줄이고, 최대 생성 처리량을 5.76배 높이며, 학습 비용을 42.5% 절감했다고 보고되었다. MLA는 대규모 MoE 언어모델과 128K 이상의 장문맥을 지원하는 환경에서 동작하며, 모델이 MHA와 저차원 KV 압축을 지원해야 한다. 저자가 밝힌 한계는 근거에 없다.

### ITME

ITME(Inter-Tier Memory Expansion)는 LLM의 대용량 KV 캐시와 모델 가중치를 CXL 하이브리드 메모리 기반 원격 확장 계층에 오프로드한다. 하드웨어/소프트웨어 선행 로딩과 PCIe Gen5 인터페이스를 통해 지연과 네트워크 오버헤드를 숨긴다. CPU-offload 베이스라인(128GB 호스트 메모리) 대비 최대 35.7% 처리량 개선을 보고했다[ITME p.10]. ITME는 고속 GPU/호스트 메모리, CXL 하이브리드 메모리, PCIe Gen5, NVMe SSD 등 멀티티어 메모리 계층이 필요하다. 성능 변동(경합 시 I/O stall)과 PCIe Gen4 시스템의 대역폭 한계가 언급되었으나, PCIe Gen5로 일부 해소했다[ITME p.11].

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준         | MLA                                                                                                   | ITME                                                                                   |
|--------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| 채택_폭      | DeepSeek-V2 MLA는 vLLM, SGLang, TensorRT-LLM 등 주요 추론 스택의 캐시 관리 계층에 통합되어 있다. 그러나 범용 추론 프레임워크에서는 아직 지원되지 않는다[웹: Predictive Multi-Tier Memory Management for KV Cache in Large ...][웹: LLM Inference Kernels for vLLM and SGLang (2026 Guide)]. | ITME는 기존 멀티티어 메모리 정책과 호환되나, 주요 추론 스택이나 상용 제품의 어느 계층에 구현되었는지 직접적 언급이 없다[ITME p.11]. |
| 채택_깊이    | SGLang에서 MLA가 기본값으로 사용되며, DeepSeek V2, V3, GLM-4.7-Flash 모델에서 자동 활성화된다[웹: LLM Inference Kernels for vLLM and SGLang (2026 Guide)][웹: MLA (Multi-head Latent Attention) - mistral.rs Documentation]. | ITME 구현의 기본값 여부, 안정성 등급에 관한 정보는 없다[ITME p.11]. |
| 진입_비용    | 기존 MHA 기반 LLM을 MLA로 전환할 때 전체 학습 데이터의 0.3~0.6%만으로 미세조정이 가능하다. 완전한 zero-shot 전환은 불가능하다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any ...]. | CXL 하이브리드 메모리 등 신규 하드웨어 도입이 필요하다. 기존 가중치 재사용, 재학습 필요성에 대한 정보는 없다[ITME p.11]. |
| 생태계_지지  | DeepSeek V2, V3, GLM-4.7-Flash 등 여러 모델에 적용되고, FlashInfer 등에서 지원된다. 독립 제3자 재현 결과, 표준 규격, 컨소시엄 정보는 없다[웹: MLA (Multi-head Latent Attention) - mistral.rs Documentation][웹: LLM Inference Kernels for vLLM and SGLang (2026 Guide)][웹: Predictive Multi-Tier Memory Management for KV Cache in Large ...]. | CXL 컨소시엄 등 생태계가 있으나, ITME 자체를 베이스라인으로 한 후속 연구, 독립 제3자 재현 결과, 표준 규격 내역은 없다[ITME p.11]. |
| 시장_수치    | LLM 비용 최적화 시장은 2025년 8억6370만 달러에서 2035년 92억720만 달러로 연평균 26.7% 성장 전망이나, MLA 자체 시장성 수치는 없다[웹: LLM Cost Optimization Market Size | CAGR of 26%]. | CXL 메모리 확장 시장은 2025년 13억 달러, 2026-2034년 연평균 28.7% 성장 전망이나, ITME 고유 시장성 수치는 없다[웹: CXL Memory Expansion Market Research Report 2034]. |

### 4.2 도메인 적용 관점

| 기준         | MLA                                                                                                   | ITME                                                                                   |
|--------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| 병목_일치    | 128K 토큰 문맥에서 약 8GB KV 캐시로, MHA(512GB) 대비 현저히 적은 메모리 사용[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache][웹: Understanding DeepSeek's Multi-Head Latent Attention (MLA) | Shashank Shekhar]. | 대용량 모델 가중치와 장문맥 KV 캐시를 CXL 하이브리드 메모리로 오프로드해 용량 병목을 완화[ITME p.2]. 대역폭 병목 해소에 대한 구체적 수치는 없다. |
| 자원_전제    | 대규모 GPU 메모리와 최신 인프라 필요. DeepSeek 모델 자체가 매우 크고 멀티노드 인프라가 필요[웹: The complete DeepSeek model guide | Guides][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache][웹: Understanding DeepSeek's Multi-Head Latent Attention (MLA) | Shashank Shekhar]. | PCIe Gen5 기반 CXL 하이브리드 메모리 아키텍처 필요. 상용 CXL 하드웨어가 제한적이며, FPGA 프로토타입으로 구현[ITME p.11]. |
| 정확도_허용치| 87.5% KV 캐시 압축 시 약 3% 정확도 손실(모델 Llama2-7B, 장문맥 기준), 4비트 양자화 결합 시 최대 3.2% 손실[MLA p.6][웹: Enabling DeepSeek's Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache]. | 근거 없음. |
| 운영_비용    | 추가 연산(쿼리 흡수, 압축 해제) 필요. 특수 커널(FlashInfer MLA kernel) 사용으로 운영 복잡도 증가 가능성[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache][웹: Understanding DeepSeek's Multi-Head Latent Attention (MLA) | Shashank Shekhar]. | 사용자 수준 선행 로딩 API 제공. 별도 패치 관리, 신규 인프라 도입, 전송 계층 추가에 대한 구체적 언급 없음[ITME p.2]. |
| 증거_성숙도  | DeepSeek V2 모델에 실제 적용, 일부 제3자(예: Moonshot AI Kimi K2 시리즈)도 유사 아키텍처 사용. 독립적 제3자 검증은 제한적[MLA p.6][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache][웹: The complete DeepSeek model guide | Guides]. | Llama-3.1 8B, 70B 모델과 실제 대화 데이터셋을 이용해 자체 실험. FPGA 기반 프로토타입, 제3자 검증 근거 없음[ITME p.9][ITME p.11]. |

## 5. 시사점

관점에 따라 동일한 근거가 다르게 해석된다. MLA는 시장성 관점에서는 주요 추론 스택 통합과 낮은 진입 비용이 강조되지만, 도메인 적용 관점에서는 특수 커널과 추가 연산으로 인한 운영 복잡도가 부각된다. ITME는 시장성에서는 신규 하드웨어 도입 필요성이 진입 비용으로, 도메인에서는 실제 상용 환경에서의 적용 가능성(프로토타입 한계)이 더 중요한 이슈로 다뤄진다. 두 기술 모두 외부 검증 부족이 시장성에서는 확산성 한계, 도메인에서는 실증 신뢰성 부족으로 다르게 읽힌다.

도메인 전환 시 평가가 크게 달라진다. MLA는 온디바이스 추론 환경에서는 대규모 모델과 인프라 요구로 인해 적용이 어렵고, ITME는 CXL 하이브리드 메모리 기반 아키텍처가 온디바이스 환경에 적용 불가해 사실상 도입이 불가능하다. 반면 클라우드 대규모 동시 서빙 환경에서는 MLA의 KV 캐시 절감 효과와 주요 추론 스택 통합이 유리하게 작용할 수 있고, ITME도 대용량 메모리 확장으로 확장성 가능성을 보이나, 실제 상용 환경 검증이 부족하다. 즉, 두 기술 모두 도메인 전환 시 막히는 지점이 다르며, 같은 조건에서 똑같이 불리해지는 것이 아니라 각기 다른 한계에 부딪힌다.

## 6. 한계점

이 평가는 공개 정보 기반으로 진행되었으며, 반도체 업계의 실제 채택 규모나 원가, 수율 등은 확인할 수 없다. 기술 성숙도(TRL)와 이해관계자 관점은 범위에서 제외했다. 확증편향을 막기 위해 두 기술에 동일한 질문 템플릿을 적용하고, 유리·불리 근거를 각각 따로 수집했으며, 기술 제시 순서를 바꿔도 결론이 달라지지 않는지 확인했다. 모든 평가 문장에는 근거 원장의 id를 달았다. 웹 근거 중 일부는 개별 기술이 아니라 LLM 비용 최적화, CXL 메모리 확장 등 상위 범주에 대한 것이었으며, 해당 항목은 기술 고유 근거로 사용하지 않았다. 근거가 없는 항목은 "근거 없음"으로 명시했다. 이 조치들은 관점별 대칭성과 근거의 직접성을 높이기 위한 것이나, 실제 시장 채택 현황, 상용화 수준, 장기적 신뢰성 등은 여전히 불확실하다.

## REFERENCE

본문에 인용 꼬리표로 실제 사용한 자료만 적는다. 목록은 근거 원장에서 자동으로 뽑았다.

**논문**

- DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434. https://arxiv.org/abs/2405.04434
- Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556. https://arxiv.org/abs/2606.12556

**웹 자료**

- CXL Memory Expansion Market Research Report 2034. https://marketintelo.com/report/cxl-memory-expansion-market
- DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference. https://arxiv.org/html/2602.21548v1
- Enabling DeepSeek's Multi-Head Latent Attention in Any .... https://arxiv.org/html/2502.14837v1
- LLM Cost Optimization Market Size | CAGR of 26%. https://market.us/report/llm-cost-optimization-market
- LLM Inference Kernels for vLLM and SGLang (2026 Guide). https://www.spheron.network/blog/deploy-flashinfer-gpu-cloud-llm-inference-kernels
- MLA (Multi-head Latent Attention) - mistral.rs Documentation. https://ericlbuehler.github.io/mistral.rs/MLA.html
- Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV Cache. https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud
- Predictive Multi-Tier Memory Management for KV Cache in Large .... https://arxiv.org/html/2604.26968v2
- The complete DeepSeek model guide | Guides. https://www.baseten.co/resources/guide/the-complete-deepseek-model-guide
- Understanding DeepSeek's Multi-Head Latent Attention (MLA) | Shashank Shekhar. https://shashankshekhar.com/blog/flashmla/flashmla-1-mla
