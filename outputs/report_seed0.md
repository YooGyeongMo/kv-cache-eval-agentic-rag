# SUMMARY

본 보고서는 장문맥 멀티턴 에이전트 서비스에서 발생하는 KV cache 병목을 해결하는 두 가지 대표적 접근, DeepSeek-V2 Multi-head Latent Attention(MLA, 소프트웨어 기반)과 ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories(ITME, 하드웨어 기반)을 비교한다. 두 기술은 진입 비용의 방향이 정반대다. MLA는 모델 구조를 바꿔 KV cache 자체를 작게 만들지만, 기존 모델을 그대로 쓸 수 없고 재학습이 필요하다. ITME는 모델을 그대로 두고 CXL 메모리 등 새로운 하드웨어 계층을 도입해 담을 공간을 넓힌다. 시장성 관점에서는 도입 장벽과 생태계 확산성, 도메인 적용 관점에서는 실제 운영 편의성과 성능 검증에 초점을 둔다. 두 관점은 각 기술의 강점과 약점을 다르게 평가하며, 도메인 전환(온디바이스 추론, 클라우드 대규모 서빙)에 따라 유불리가 뒤바뀐다.

## 1. 분석 배경

장문맥 멀티턴 에이전트 서비스는 단일 세션에서 100K 토큰을 넘는 긴 문맥을 유지하며, 대화가 이어질수록 KV cache가 턴마다 누적된다. 이때 KV cache의 크기가 기하급수적으로 커져 GPU 메모리 용량과 대역폭이 병목이 된다. 기존 Multi-Head Attention(MHA) 구조에서는 각 토큰마다 head 수와 head dimension에 비례해 KV cache가 쌓이므로, 장문맥 환경에서는 동시 세션 수가 제한되고, 비용이 급증한다[MLA p.6][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV].

이 병목을 해결하는 접근은 크게 두 갈래로 나뉜다. 하나는 소프트웨어적으로 KV cache 자체를 작게 만드는 방식이다. MLA는 어텐션 구조를 바꿔, 처음부터 KV cache를 저차원(latent)으로 압축해 저장한다. 이 방식은 하드웨어를 그대로 두고 모델 구조를 바꾼다. 다른 하나는 하드웨어적으로 담을 공간을 넓히는 방식이다. ITME는 CXL 기반 하이브리드 메모리 계층을 도입해, 대용량 KV cache와 모델 가중치를 원격 확장 계층에 오프로드한다. 이 방식은 모델을 그대로 두고 시스템 인프라를 바꾼다[ITME p.2][ITME p.11].

두 접근은 진입 비용의 방향이 정반대다. MLA는 기존 모델을 그대로 쓸 수 없고, MLA 구조로 재학습이 필요하다. ITME는 기존 모델을 그대로 쓸 수 있지만, 새로운 하드웨어 인프라 도입이 필요하다. 이 때문에 어느 쪽이 유리하다고 단정할 수 없고, 관점에 따라 평가가 갈린다. 본 보고서는 이 대칭성을 바탕으로 두 기술을 시장성, 도메인 적용 관점에서 대조한다.

## 2. 기술 선정


기술 선정은 2안으로 했다. 에이전트가 고르는 방식은 쓰지 않고 사람이 문서 풀에서 진영별로 하나씩 직접 골랐다.

두 기술을 고른 공통 기준은 진입 비용의 대칭성. 한쪽은 하드웨어를 그대로 두는 대신 모델 쪽을 바꿔야 하고, 다른 쪽은 모델을 그대로 두는 대신 새 메모리 하드웨어를 들여야 한다. 같은 병목을 두고 무엇을 바꿀 것인지가 정반대라서, 우열을 판정하지 않고도 관점에 따라 평가가 갈리는 모습을 드러낼 수 있다.

MLA (소프트웨어, 데이터를 작게 만들기): 소프트웨어 진영에서 가장 근본적인 형태를 고른다. 사후 압축은 이미 만들어진 KV cache를 줄이지만 MLA는 어텐션 구조 자체를 바꿔 처음부터 작게 나오게 한다. 압축률 93.3퍼센트[MLA p.1]로 진영 안에서 가장 큰 수치를 보고했고, 실제 상용 모델에 배포되어 시장 근거를 모을 수 있다. 진입 비용을 볼 때는 기법 자체와 후속 연구를 구별해야 한다. MLA는 그 구조로 사전학습한 모델을 전제하므로 기존 모델에 그대로 얹을 수 없고, 기존 모델을 MLA로 바꾸려면 TransMLA 같은 별도 전환 연구를 거쳐야 한다.

ITME (하드웨어, 담을 공간을 넓히기): 하드웨어 진영에서 분류가 흔들리지 않는 것을 고른다. 같은 진영의 InfiniGen은 기존 하드웨어 위에서 도는 소프트웨어 기법이라 진영 구분이 흐려진다. ITME는 CXL 메모리라는 새 하드웨어 계층을 전제하므로 진입 비용의 성격이 MLA와 정확히 대비되고, 멀티턴 워크로드를 정면으로 측정해 주 도메인과도 맞는다.

같은 문서 풀에서 뺀 후보와 그 이유는 다음과 같다.

TurboQuant의 경우 사후 압축 계열. MLA와 같은 진영이라 한 진영에서 둘을 고를 수 없다.

KIVI의 경우 사후 압축 계열. 위와 같다.

InfiniGen의 경우 기존 하드웨어 위에서 도는 소프트웨어 기법이라 하드웨어 진영 대표로 두면 비교 축이 흐려진다.

PIM/CXL의 경우 수치는 가장 크지만 데이터센터급 PIM의 상용 채택 근거가 확인되지 않아 시장성 관점에서 수집할 근거가 한쪽으로 빈다.


## 3. 기술 개요

### MLA

MLA(Multi-head Latent Attention)는 어텐션 구조에서 key와 value를 저차원(latent) 벡터로 공동 압축해, 추론 시 KV cache 크기를 크게 줄이는 기법이다. 기존 Multi-Head Attention(MHA) 대비 93.3%의 KV cache 감소(DeepSeek 67B 기준)[MLA p.1], 최대 5.76배의 생성 처리량 증가, 42.5%의 학습 비용 절감이 보고되었다(모두 DeepSeek 67B와 비교)[MLA p.1]. MLA는 Transformer 기반 모델 구조에서 동작하며, 최대 128K 토큰의 긴 문맥을 지원한다. 이 기법은 attention 메커니즘 내부에서 key-value를 저차원으로 압축하는 구조를 전제로 한다.

MLA의 성립 조건은 Transformer 기반 모델이어야 하며, multi-head attention 구조를 지원해야 한다. 또한, low-rank key-value joint compression이 attention 메커니즘에 내장되어야 한다. 저자가 밝힌 한계에 대한 직접적 근거는 없다.

### ITME

ITME(Inference Tiered Memory Expansion)는 LLM의 대용량이면서 예측 가능성이 높은 모델 가중치와 장기 컨텍스트 KV 캐시를 원격 확장 계층(CXL 하이브리드 메모리)으로 오프로드한다. 하드웨어와 소프트웨어가 협력하는 프리페칭 전략을 통해 NVMe SSD에서 CXL 하이브리드 메모리의 내부 DRAM 캐시로 데이터를 미리 불러와 저장 및 네트워크 지연을 완화한다. 활성화 및 작업 KV 캐시는 고속 GPU 또는 호스트 메모리 계층에 배치해 지연에 민감한 데이터를 처리한다[ITME p.2][ITME p.11].

ITME의 성립 조건은 CXL 하이브리드 메모리를 활용한 다계층 메모리 시스템, 고속 GPU 및 호스트 메모리 계층과 원격 확장 계층 간의 협력적 데이터 배치, 하드웨어 수준 프리페칭과 NVMe-oF 스토리지 액세스 지원 환경이다. 저자가 밝힌 한계에 대한 직접적 근거는 없다.

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준         | MLA                                                                                                                       | ITME                                                                                                         |
|--------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| 채택 폭      | DeepSeek-V2 MLA는 vLLM, SGLang, TensorRT-LLM 등 주요 추론 스택에서 지원되나, 범용 프레임워크에서는 완전한 통합이 미흡하다. H200, B200 GPU에서 MLA 전용 커널이 활용된다[웹: Predictive Multi-Tier Memory Management for KV Cache][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Enabling DeepSeek's Multi-Head Latent Attention in Any][웹: Towards Economical Inference: Enabling DeepSeek's]. | ITME는 FlexGen, DeepSpeed Inference 등 기존 소프트웨어 스택과 호환되도록 설계되었으나, 주요 추론 스택이나 상용 제품에 실제 구현되었다는 근거는 없다[ITME p.11][ITME p.1]. |
| 채택 깊이    | vLLM에서는 DeepSeek 계열 모델에 대해 MLA가 기본값으로 활성화된다. SGLang에서는 플래그로 활성화 가능하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Towards Economical Inference: Enabling DeepSeek's]. | ITME 구현이 기본값인지, 선택 옵션인지, 안정성 등급 등은 문서에 명시되어 있지 않다[ITME p.11][ITME p.1]. |
| 진입 비용    | 기존 MHA 또는 GQA 기반 LLM은 MLA 아키텍처와 호환되지 않아, MLA 도입을 위해서는 MHA2MLA 등 데이터 효율적 파인튜닝 프레임워크로 재학습이 필요하다. 기존 가중치 재사용은 불가능하다. MLA 전용 커널을 활용하려면 특정 GPU(H200, B200) 및 플래그 설정이 필요하다[웹: Towards Economical Inference: Enabling DeepSeek's][웹: Predictive Multi-Tier Memory Management for KV Cache][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | ITME는 CXL-hybrid 메모리와 PCIe Gen5 인터페이스를 요구한다. 기존 가중치는 그대로 사용하며, 하드웨어 및 소프트웨어 협조적 프리페칭 전략을 통해 접근 지연을 완화한다. 구체적 재학습 필요성이나 하드웨어 교체 범위에 대한 명확한 언급은 없다[ITME p.2][ITME p.11]. |
| 생태계 지지  | DeepSeek가 제안한 혁신적 아키텍처로, MHA2MLA 후속 연구가 존재한다. 표준 규격, 컨소시엄, 독립 제3자 재현 결과 공개에 대한 근거는 없다[웹: Towards Economical Inference: Enabling DeepSeek's]. | CXL 기술은 여러 연구와 컨소시엄에서 다뤄지나, ITME 자체를 베이스라인으로 삼은 후속 연구, 독립 제3자 재현 결과, 표준 규격 및 컨소시엄 참여에 관한 직접적 근거는 없다[ITME p.11]. |
| 시장 수치    | LLM 시장 규모는 2025년 6.48 billion USD에서 2034년 95.90 billion USD로 성장 전망(CAGR 34.90%). 이 수치는 LLM 상위 범주에 대한 전망이며 MLA 기술 자체를 직접 지칭하지 않는다[웹: LLM Market Size to Hit USD 95.90 Billion by 2034 /]. | CXL 메모리 확장 시장은 2026년 1.74 billion 달러로 전망되며, 이 수치는 CXL 메모리 확장 상위 범주를 가리킨다. ITME 기술 자체에 대한 직접적 시장 수치는 없다[웹: CXL Memory Expansion Market Size & Share Report]. |

### 4.2 도메인 적용 관점

| 기준         | MLA                                                                                                                       | ITME                                                                                                         |
|--------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| 병목 일치    | 장문맥 멀티턴 에이전트 서비스에서 KV cache가 메모리 용량과 대역폭 병목의 핵심이며, MLA는 KV cache 크기를 93.3% 줄여 이 병목과 일치하는 해결책을 제시한다[MLA p.6][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV].[MLA p.1] | ITME는 대용량 모델 가중치와 장문맥 KV 캐시를 원격 확장 계층에 오프로드해, 용량과 예측 가능성 기반 병목 해소 전략과 일치한다[ITME p.2]. |
| 자원 전제    | 128 attention heads, head_dim=128, d_c=512 설정에서 토큰당 KV cache를 1,024 bytes로 줄여, H200, B200 GPU에서 8-12배 더 많은 동시 사용자 처리가 가능하다. CUDA 12.4+ 환경과 vLLM, FlashInfer 등 소프트웨어 지원이 필요하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Build DeepSeek-V3: Multi-Head Latent Attention (MLA)]. | PCIe Gen5 인터페이스를 사용하는 CXL-하이브리드 메모리 아키텍처를 전제로 하며, GPU 및 호스트 메모리 계층과의 협업이 필요하다[ITME p.11][ITME p.2]. |
| 정확도 허용치| 근거 없음  측정은 LongBench 장문맥 과제에서 이루어졌다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any]. 100K 토큰 이상의 단일 세션 문맥과 멀티턴 대화 환경에서의 정확도 손실에 대한 직접적 수치는 없다. | ITME의 정확도 손실 수치나 측정 조건에 관한 직접적 보고는 없다. 장문맥 KV 캐시 오프로드가 정확도에 미치는 영향에 대한 구체적 수치도 제시되지 않았다[ITME p.2]. |
| 운영 비용    | MLA 도입 시 별도의 플래그 없이 vLLM이 모델 config에서 자동 인식하며, H200, B200 GPU 인스턴스와 CUDA 12.4+ 환경에서 운영 가능하다. 별도의 전송 계층 추가 없이 KV cache 크기 감소로 GPU 비용이 8-9배 절감되나, 스팟 인스턴스 특성상 SLA 보장을 위한 추가 계획이 필요하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | ITME는 하드웨어 수준의 프리페칭과 사용자 수준 API를 제공하여 데이터 전송을 최적화하며, 기존 소프트웨어 정책과 호환되어 별도의 소프트웨어 스택 교체 없이 통합 가능하다. CXL 하드웨어 도입과 PCIe Gen5 기반 인프라 구축, 프리페칭 제어를 위한 신규 API 도입이 필요하다[ITME p.2][ITME p.11]. |
| 증거 성숙도  | MLA 성능과 효율성 수치는 DeepSeek 팀의 자체 보고 및 논문에서 직접 측정한 결과이며, 일부는 DeepSeek-AI 공식 문서와 논문에서 일차 자료로 제공된다[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in]. | ITME는 FPGA 프로토타입과 생산급 SK hynix CMM을 이용한 성능 평가를 수행했으며, 일부 수치는 저자 자체 보고에 기반한다. 제3자 검증이나 대규모 실제 서비스 적용 사례는 아직 보고되지 않았다[ITME p.2][ITME p.11]. |

## 5. 시사점

두 기술은 진입 비용의 방향이 정반대다. MLA는 기존 모델을 그대로 쓸 수 없고, MLA 구조로 재학습이 필요하다. 이 점은 시장성 관점에서 도입 장벽으로 작용한다[웹: Towards Economical Inference: Enabling DeepSeek's]. 반면, 도메인 적용 관점에서는 vLLM 등 주요 추론 스택에서 별도 플래그 없이 자동 활성화되어 운영 복잡도가 낮다고 평가된다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. 시장성은 초기 전환 비용에, 도메인은 실운영 편의성에 초점을 둔다.

생태계 및 표준화 기반에서도 관점이 엇갈린다. 시장성 관점은 표준 규격, 컨소시엄, 제3자 재현 결과 등 외부 확산성과 신뢰 기반을 중시한다[웹: Towards Economical Inference: Enabling DeepSeek's]. 도메인 적용 관점은 공식 논문과 자체 실험 결과 등 일차 자료의 신뢰성을 근거로 삼는다[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in].

ITME의 경우, 시장성 관점에서는 실제 상용화 및 구현 사례의 부재를 도입의 불리함으로 본다[ITME p.11][ITME p.1]. 도메인 적용 관점에서는 설계상 호환성과 프리페칭 전략의 운영상 이점을 강조한다[ITME p.2][ITME p.11]. 시장성은 실증적 채택 현황, 도메인은 설계 및 이론적 호환성에 초점을 둔다.

정확도 및 성능 검증에서도 차이가 있다. 시장성 관점은 외부 검증과 표준화, 후속 연구의 존재를 신뢰성의 핵심으로 본다[ITME p.11]. 도메인 관점은 실제 서비스 환경에서의 정확도 손실 수치 부재를 문제로 본다[ITME p.2].

도메인을 바꾸면 평가가 뒤집힌다. 온디바이스 추론에서는 MLA가 고성능 GPU와 CUDA 12.4+ 환경을 요구해 적용이 제한적이다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. ITME는 CXL 하드웨어가 없는 스마트폰, 엣지 디바이스에서는 적용이 불가능하다[ITME p.11]. 클라우드 대규모 동시 서빙에서는 MLA가 KV cache 크기를 줄여 동시 세션 처리에 유리하게 평가되고[MLA p.6][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV], ITME는 메모리 확장이 중요한 환경에서 유리하게 평가된다. 단, ITME는 상용 CXL 하드웨어 가용성과 인프라 구축 비용이 추가 고려사항이 된다[ITME p.2][ITME p.11].

## 6. 한계점

이 평가는 1인 과제 범위로 한정했다. 기술 성숙도(TRL) 관점과 이해관계자 관점은 범위에서 뺐다. 빠뜨린 것이 아니라 뺀 것이다. 두 기술의 공개 시점이 2024년 6월과 2026년 6월로 2년 차이 나므로, 채택 현황을 나란히 놓으면 늦게 나온 쪽이 불리하게 읽힌다. 이 시간 차이를 감안하고 읽어야 한다.

시장 근거는 공개 자료만 썼다. 반도체 업계는 수율과 원가를 공개하지 않으므로 실제 채택 규모는 공개 정보로 확인할 수 없다. 웹 근거에는 출처 등급을 붙였다. MLA: 일차 7건, 이차 49건, 시장조사 0건. ITME: 일차 14건, 이차 42건, 시장조사 0건이다. 시장 규모 수치는 대부분 시장조사 등급이며, 원본 보고서를 열어 대조하지 못했다. 같은 범주를 두고도 조사기관마다 수치가 크게 다른 경우가 있으므로 단일 수치를 그대로 믿으면 안 된다.

확증편향을 막기 위해 두 기술에 같은 질문 템플릿을 썼고, 관점마다 유리한 근거와 불리한 근거를 각각 따로 검색했다. 기술 제시 순서를 바꿔 두 번 실행하고, 결론이 달라지는지 기록했다. 모든 평가 문장에는 근거 원장의 id를 달았고, 수치는 근거 원문에 있는지 코드로 대조했다. 우열을 판정하는 표현은 금지어 목록으로 검사했다. 이 조치들은 한쪽으로 기운 평가를 줄이지만, 근거가 부족한 항목이나 범주 수준의 수치(예: LLM 시장 규모, CXL 시장 규모)는 개별 기술의 실제 채택이나 효과를 직접 보여주지 않는다. 웹 근거가 개별 기술이 아니라 범주 수준이었던 항목은 시장성 수치와 일부 생태계 지지 항목이다.

> 이 보고서는 생성 후 검증 단계를 거쳤다. 인용 꼬리표가 가리키는 근거에 해당 내용이 없는 문장은 그 내용이 실제로 나온 출처로 꼬리표를 바로잡았고, 문서 풀 어디에도 근거가 없는 항목은 근거 없음으로 내렸다. 참고문헌의 링크는 접속을 확인했고 실패한 것은 그 사실을 적었다.

## REFERENCE

본문에 인용 꼬리표로 실제 사용한 자료만 적는다. 목록은 근거 원장에서 자동으로 뽑았다.

**논문**

- DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434. https://arxiv.org/abs/2405.04434
- Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556. https://arxiv.org/abs/2606.12556

**웹 자료**

- Pyimagesearch(작성일 미상). *Build DeepSeek-V3: Multi-Head Latent Attention (MLA)*. pyimagesearch.com, https://pyimagesearch.com/2026/03/16/build-deepseek-v3-multi-head-latent-attention-mla-architecture
- Snsinsider(작성일 미상). *CXL Memory Expansion Market Size & Share Report*. snsinsider.com, https://www.snsinsider.com/reports/cxl-memory-expansion-market-10881
- arXiv(작성일 미상). *DualPath: Breaking the Storage Bandwidth Bottleneck in*. arxiv.org, https://arxiv.org/html/2602.21548v1
- ACL Anthology(작성일 미상). *Enabling DeepSeek's Multi-Head Latent Attention in Any*. aclanthology.org, https://aclanthology.org/2025.acl-long.1597.pdf
- Trendxinsights(작성일 미상). *LLM Market Size to Hit USD 95.90 Billion by 2034 /*. trendxinsights.com, https://trendxinsights.com/syndicated-market-research-reports/llm-market
- Spheron Network(작성일 미상). *Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV*. spheron.network, https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud
- arXiv(작성일 미상). *Predictive Multi-Tier Memory Management for KV Cache*. arxiv.org, https://arxiv.org/html/2604.26968v2
- Alphaxiv(작성일 미상). *Towards Economical Inference: Enabling DeepSeek's*. alphaxiv.org, https://www.alphaxiv.org/abs/2502.14837
