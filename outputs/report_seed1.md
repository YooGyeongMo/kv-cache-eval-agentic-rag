# SUMMARY

이 보고서는 장문맥 멀티턴 에이전트 서비스에서 발생하는 KV cache 병목을 해소하기 위한 두 가지 대표적 접근법, ITME(CXL-hybrid 메모리 기반 하드웨어 확장)와 MLA(Multi-head Latent Attention 기반 소프트웨어 구조 변경)를 대조한다. 두 기술은 진입 비용, 적용 범위, 시장성, 도메인 적합성 등에서 정반대의 특성을 보인다. ITME는 기존 모델을 그대로 두고 하드웨어를 확장하는 방식이며, MLA는 하드웨어를 그대로 두고 모델 구조를 바꿔야 한다. 시장성 관점에서는 도입 장벽과 생태계 확산성, 도메인 관점에서는 실제 서비스 환경에서의 성능과 운영 편의성에 따라 평가가 갈린다. 도메인 전환 시 두 기술의 한계가 서로 다르게 드러난다. 우열을 판정하지 않고 관점별로 평가가 어떻게 달라지는지에 초점을 맞췄다.

## 1. 분석 배경

장문맥 멀티턴 에이전트 서비스는 단일 세션에서 100K 토큰이 넘는 문맥을 유지하며, 대화가 여러 턴에 걸쳐 이어진다. 이 과정에서 각 턴마다 KV cache가 누적되어 GPU 메모리 용량과 대역폭, 지연, 전력 측면에서 병목이 발생한다. 기존의 Multi-Head Attention(MHA) 구조에서는 KV cache가 선형적으로 증가해, 장문맥 환경에서는 메모리 사용량이 급격히 커진다. 이로 인해 대규모 모델의 실시간 추론, 동시 세션 처리, 비용 효율적 운영이 어려워진다.

이 병목을 해소하는 접근은 크게 두 갈래로 나뉜다. 첫째는 하드웨어 계층에서 메모리 용량을 확장하는 방식이다. 대표적으로 ITME는 CXL-hybrid 메모리와 PCIe Gen5 인터페이스를 활용해 GPU 서버의 메모리 용량을 원격 확장 계층까지 넓힌다. 이 방식은 기존 모델과 소프트웨어 스택을 그대로 두고, 하드웨어 인프라를 추가하는 것이 특징이다. 둘째는 소프트웨어 계층에서 KV cache 자체를 작게 만드는 방식이다. MLA는 어텐션 구조를 변경해 처음부터 KV cache가 작게 생성되도록 설계한다. 이 방식은 하드웨어를 그대로 두고, 모델 구조와 학습 과정을 바꿔야 한다.

두 접근은 진입 비용의 방향이 정반대다. ITME는 새 하드웨어 도입이 필요하지만 모델 재학습이 필요 없고, MLA는 기존 하드웨어를 그대로 쓰지만 모델을 새로 학습해야 한다. 이 대칭성 때문에 어느 쪽이 더 낫다고 단정할 수 없으며, 관점에 따라 평가가 달라진다. 시장성 관점에서는 도입 장벽과 생태계 확산성, 도메인 관점에서는 실제 워크로드에서의 성능과 운영 편의성이 주요 평가 기준이 된다.

## 2. 기술 선정

이 보고서에서 다루는 두 기술은 ITME와 MLA다. 선정은 에이전트가 아니라 사람이 Doc Pool에서 진영별로 하나씩 직접 골랐다.

공통 선정 기준은 진입 비용의 대칭성이다. 한쪽은 하드웨어를 그대로 두는 대신 모델 쪽을 바꿔야 하고, 다른 쪽은 모델을 그대로 두는 대신 새 메모리 하드웨어를 들여야 한다. 같은 병목을 두고 무엇을 바꿀 것인지가 정반대라서, 우열을 판정하지 않고도 관점에 따라 평가가 갈리는 모습을 드러낼 수 있다.

MLA(소프트웨어, 데이터를 작게 만들기)는 소프트웨어 진영에서 가장 근본적인 형태를 고른다. 사후 압축은 이미 만들어진 KV cache를 줄이지만 MLA는 어텐션 구조 자체를 바꿔 처음부터 작게 나오게 한다. 압축률 93.3퍼센트[MLA p.1]로 진영 안에서 가장 큰 수치를 보고했고, 실제 상용 모델에 배포되어 시장 근거를 모을 수 있다. 진입 비용을 볼 때는 기법 자체와 후속 연구를 구별해야 한다. MLA는 그 구조로 사전학습한 모델을 전제하므로 기존 모델에 그대로 얹을 수 없고, 기존 모델을 MLA로 바꾸려면 TransMLA 같은 별도 전환 연구를 거쳐야 한다.

ITME(하드웨어, 담을 공간을 넓히기)는 하드웨어 진영에서 분류가 흔들리지 않는 것을 고른다. 같은 진영의 InfiniGen은 기존 하드웨어 위에서 도는 소프트웨어 기법이라 진영 구분이 흐려진다. ITME는 CXL 메모리라는 새 하드웨어 계층을 전제하므로 진입 비용의 성격이 MLA와 정확히 대비되고, 멀티턴 워크로드를 정면으로 측정해 주 도메인과도 맞는다.

같은 Doc Pool에서 뺀 후보와 그 이유는 다음과 같다. TurboQuant와 KIVI는 사후 압축 계열로 MLA와 같은 진영이라 한 진영에서 둘을 고를 수 없다. InfiniGen은 기존 하드웨어 위에서 도는 소프트웨어 기법이라 하드웨어 진영 대표로 두면 비교 축이 흐려진다. PIM/CXL은 수치는 가장 크지만 데이터센터급 PIM의 상용 채택 근거가 확인되지 않아 시장성 관점에서 수집할 근거가 한쪽으로 빈다.

임베딩 모델(bge-m3) 선정은 검색 설정이지 기술 선정과는 무관하다.

## 3. 기술 개요

### ITME

ITME는 KV cache 병목을 해결하기 위해 CXL-hybrid 메모리 아키텍처를 활용한다. 대용량, 예측 가능한 데이터(모델 가중치, 장문맥 KV cache 등)를 원격 확장 계층에 오프로드하고, NVMe SSD에서 DRAM 캐시로 하드웨어 수준 프리페칭을 수행한다. HW/SW 협업 프리페칭 전략을 통해 저장장치 및 네트워크 지연을 가린다. GPU 서버는 표준 RDMA를 통해 대용량 모델 가중치와 KV cache에 접근할 수 있다[ITME p.2][ITME p.11].

보고된 수치는 다음과 같다. CPU-offload 베이스라인에서 128 GB host memory(T2)를 캐싱에 사용하며, 35턴 벤치마크와 256개 동시 대화(ShareGPT 데이터셋 기준)에서 측정했다. ITME는 CXL-hybrid memory(T3.5) 계층을 활용하면서 host memory(T2)는 스테이징 버퍼로만 일부 사용한다(동일 벤치마크 조건)[ITME p.11].

성립 조건은 다음과 같다. 고속 GPU 또는 host memory 계층(활성화, working KV cache용)과, 대용량 모델 가중치 및 장문맥 KV cache를 위한 원격 CXL-hybrid memory 계층이 필요하다. NVMe SSD에서 DRAM 캐시로의 하드웨어 프리페칭, PCIe Gen5 인터페이스, 사용자 수준 프리페처 API, LLM 추론 프레임워크와의 통합이 전제된다[ITME p.2][ITME p.11].

저자가 밝힌 한계는 근거에 없다.

### MLA

MLA(Multi-head Latent Attention)는 추론 시 발생하는 KV cache 병목을 완화하기 위해, Key와 Value를 저차원 잠재 벡터로 결합해 저장하는 구조다. 기존 Multi-Head Attention(MHA) 대비 훨씬 작은 KV cache로 동일하거나 더 나은 성능을 달성하도록 설계됐다[MLA p.1].

보고 수치는 다음과 같다. DeepSeek 67B 모델 대비 93.3% KV cache 감소, 42.5% 학습 비용 절감, 최대 5.76배 생성 처리량 증가, 21B 활성 파라미터로 오픈소스 모델 중 상위권 성능을 달성했다[MLA p.1].

성립 조건은 Transformer 기반 모델 구조와 multi-head attention 메커니즘 지원이 필요하다. 최대 128K 토큰 문맥 길이를 지원하며, 어텐션 모듈 내에서 low-rank key-value joint compression이 가능해야 한다[MLA p.1].

저자가 밝힌 한계는 근거에 없다.

## 4. 관점별 평가

### 4.1 시장성 관점

| 평가 기준   | ITME | MLA |
|-------------|------|-----|
| 채택_폭 | 기존 LLM 추론 소프트웨어 스택을 대체하지 않고 상호 보완하는 아키텍처로, 커널, 서빙 엔진, SDK, 하드웨어 제품 중 어느 계층에 구현되었는지에 대한 구체적 언급은 없다[ITME p.11]. | vLLM, SGLang, TensorRT-LLM 등 주요 추론 스택에서 커널 및 서빙 엔진 수준으로 통합되어 있으나, 범용 프레임워크에서는 아직 지원되지 않아 MHA 크기와 동일한 메모리 할당을 강제받는다[웹: Predictive Multi-Tier Memory Management for KV Cache][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. |
| 채택_깊이 | ITME 구현이 기본값인지 선택 옵션인지, 안정성 등급이나 기본값 여부에 관한 정보는 없다[ITME p.11]. | vLLM에서는 DeepSeek 계열 모델에 대해 MLA가 기본값으로 자동 처리되며, SGLang에서는 플래그를 통해 MLA 커널을 활성화하는 선택 옵션으로 제공된다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. |
| 진입_비용 | CXL-hybrid 메모리를 활용해 TB 규모 메모리 확장을 제공하며, 기존 가중치를 그대로 사용하고 재학습 필요성에 대한 언급은 없으나, 새 하드웨어(PCIe Gen5 기반 CXL-hybrid 메모리) 도입이 필요하다[ITME p.1][ITME p.11]. | 기존 MHA 또는 GQA 기반의 잘 훈련된 LLM은 MLA 아키텍처와 호환되지 않아, MHA2MLA라는 데이터 효율적 미세조정 프레임워크를 통해 재학습이 필요하다. 새 하드웨어 구매에 대한 직접 언급은 없다[웹: Towards Economical Inference: Enabling DeepSeek's]. |
| 생태계_지지 | CXL 기반 메모리 확장 연구의 연장선에 있으나, 후속 연구가 ITME를 베이스라인으로 삼았다는 정보, 표준 규격이나 컨소시엄 참여, 독립 제3자 재현 결과 공개는 없다[ITME p.11]. | DeepSeek가 제안한 혁신적 아키텍처로, 후속 연구에서 MHA2MLA 미세조정 방법이 발표되었으나, 표준 규격이나 컨소시엄, 독립 제3자 재현 결과 공개에 대한 언급은 없다[MLA p.6][웹: Towards Economical Inference: Enabling DeepSeek's]. |
| 시장_수치 | CXL 메모리 확장 시장은 2026년 1.74 billion 달러, 2035년 16.55 billion 달러 규모로 연평균 성장률 28.5%를 보인다. 이는 CXL 메모리 확장 상위 범주 수치로 ITME 기술 고유 수치는 아니다[웹: CXL Memory Expansion Market Size & Share Report]. | LLM 시장 규모는 2025년 6.48 billion 달러에서 2034년 95.90 billion 달러로 성장할 전망이며 CAGR은 34.90%이다. 이 수치는 LLM 범주 전체를 가리키며 MLA 기술 자체에 대한 직접적 시장 수치는 없다[웹: LLM Market Size to Hit USD 95.90 Billion by 2034 /]. |

유리한 점과 불리한 점은 다음과 같다.

ITME는 대규모 LLM 추론을 위한 비용 효율적 TB 규모 메모리 확장을 제공하며, PCIe Gen5 인터페이스를 채택해 높은 대역폭을 지원한다[ITME p.1][ITME p.11]. 기존 소프트웨어 스택과 호환되며, 다계층 KV 캐시 및 프리페칭과 시너지를 낼 수 있다[ITME p.11]. CXL 메모리 확장 시장이 빠르게 성장 중이며, AI 추론용 메모리 확장 수요가 증가하고 있다[웹: CXL Memory Expansion Market Size & Share Report]. 반면, ITME가 주요 추론 스택이나 상용 제품의 어느 계층에 구현되었는지 구체적 정보가 없으며, 기본값인지 선택 옵션인지도 불명확하다[ITME p.11]. 도입에 PCIe Gen5 기반 CXL-hybrid 메모리 등 신규 하드웨어가 필요해 진입 비용이 존재한다[ITME p.11]. ITME를 베이스라인으로 한 후속 연구, 표준화, 독립 재현 결과 등 생태계 지지 근거가 없다[ITME p.11]. 시장 수치는 CXL 메모리 확장 상위 범주에 대한 것으로, ITME 기술 고유의 시장성 수치는 없다[웹: CXL Memory Expansion Market Size & Share Report].

MLA는 vLLM에서 DeepSeek 계열 모델에 대해 MLA가 기본값으로 자동 활성화되어 사용 편의성이 높다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. 기존 MHA 대비 KV 캐시 메모리를 93.3%까지 줄여 추론 효율을 크게 개선하는 혁신적 아키텍처로 평가된다[MLA p.6][웹: Decoding Multi-Head Latent Attention (Part 1): The KV]. 반면, 범용 추론 프레임워크에서는 MLA를 지원하지 않아 MHA 크기와 동일한 메모리 할당을 강제받아 도입 장벽이 존재한다[웹: Predictive Multi-Tier Memory Management for KV Cache]. 기존 MHA 또는 GQA 기반 LLM은 MLA 적용을 위해 재학습이 필요하며, 완전한 사전학습 없이 전환이 불가능해 진입 비용이 높다[웹: Towards Economical Inference: Enabling DeepSeek's]. MLA 관련 표준 규격, 컨소시엄, 독립 제3자 재현 결과 공개 등 생태계 지지 근거는 없다[MLA p.6][웹: Towards Economical Inference: Enabling DeepSeek's]. 시장 수치는 LLM 범주 전체에 대한 전망이며 MLA 기술 자체의 시장 규모나 성장 전망 수치는 없다[웹: LLM Market Size to Hit USD 95.90 Billion by 2034 /].

### 4.2 도메인 적용 관점

| 평가 기준   | ITME | MLA |
|-------------|------|-----|
| 병목_일치 | 장문맥 멀티턴 에이전트 서비스에서 병목은 메모리 용량과 대역폭, 지연, 전력이며, ITME는 대용량 모델 가중치와 장문맥 KV 캐시를 원격 확장 계층에 오프로드하여 접근 지연을 숨기는 데 초점을 맞춘다. 이는 대규모 KV 캐시의 용량과 예측 가능성에 기반한 병목 해소 전략과 일치한다[ITME p.2][웹: HyMCache][웹: Using CXL Fabric-Attached Memory]. | 장문맥 멀티턴 에이전트 서비스에서 KV cache가 메모리 용량과 대역폭 병목의 핵심이며, MLA는 KV cache 크기를 대폭 줄여 이 병목과 일치하는 해결책을 제시한다[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. |
| 자원_전제 | PCIe Gen5 인터페이스를 사용하는 CXL-하이브리드 메모리 아키텍처를 전제로 하며, GPU 및 호스트 메모리 계층과의 협업을 전제로 한다. 다중 GPU 환경에서의 KV 캐시 교체를 지원한다[ITME p.11][ITME p.2][웹: CXL 4.0 Infrastructure Planning Guide]. | 128 attention heads, head_dim=128, d_c=512 설정에서 토큰당 1,024바이트를 저장하며, 128K 문맥, 60개 층 기준으로 7.5GB KV cache가 필요해 고성능 GPU(H200, B200)에서 다중 동시 사용자 지원이 가능하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Build DeepSeek-V3: Multi-Head Latent Attention (MLA)]. |
| 정확도_허용치 | ITME의 정확도 손실 수치나 측정 조건에 관한 직접적인 보고는 없으며, KV 캐시 오프로드가 정확도에 미치는 영향에 대한 구체적 수치도 제시되지 않았다. 다른 연구에서 KV 오프로드가 문맥 집약적 작업에서 성능 저하를 보인다는 보고가 있으나 이는 ITME 고유의 수치는 아니다[웹: KV Cache Offloading for Context-Intensive Tasks]. | MLA는 7B Llama2 모델 대비 87.5% KV cache 감소 시 3% 정확도 손실을 보이며, 4-bit 양자화와 결합 시 최대 96.87% 압축에 -0.5%~ -3.2% 정확도 손실 범위 내에 있다. 측정은 LongBench 장문맥 과제에서 이루어졌다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any]. |
| 운영_비용 | 사용자 수준 프리페처 API를 제공하여 LLM 추론 프레임워크가 데이터 전송을 명시적으로 제어할 수 있게 하며, PCIe 대역폭을 최대한 활용하는 스케줄링을 지원한다. 별도의 패치 관리, 신규 인프라 도입, 전송 계층 추가 등에 대한 구체적 언급은 없다[ITME p.2]. | vLLM과 SGLang에서 네이티브 지원되어 별도 플래그 없이 활성화 가능하며, 추가 전송 계층이나 패치 관리 없이 기존 GPU 인프라에서 운영 가능하다. H200 GPU에서 KV cache FP4 경로 부재로 소프트웨어 에뮬레이션이 발생할 수 있다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. |
| 증거_성숙도 | SK hynix CMM과 Gen5 PCIe 기반 FPGA 프로토타입을 사용해 성능 잠재력을 검증했으며, 일부 수치는 저자 자체 보고에 기반한다. 상용 CXL 하드웨어의 제한으로 인해 시뮬레이션과 프로토타입 평가가 주를 이루며, 제3자 검증은 근거에 나타나지 않는다[ITME p.11][ITME p.9]. | MLA 성능과 효율성 수치는 DeepSeek 저자 자체 보고와 DeepSeek-AI 공식 문서, 일부 제3자 검증(일차 출처) 및 이차 출처에서 측정된 실제 장문맥 멀티턴 에이전트 워크로드 환경 기반이다[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in][웹: Enabling DeepSeek's Multi-Head Latent Attention in Any]. |

유리한 점과 불리한 점은 다음과 같다.

ITME는 장문맥 KV 캐시와 모델 가중치의 예측 가능성과 대용량 특성을 활용해 원격 확장 계층에 오프로드함으로써 메모리 용량 병목을 효과적으로 다룬다[ITME p.2]. PCIe Gen5 인터페이스를 채택한 CXL-hybrid 메모리 아키텍처로 고대역폭을 제공하여 다중 GPU 환경에서 KV 캐시 교체를 빠르게 수행할 수 있다[ITME p.11]. 사용자 수준 프리페처 API를 통해 LLM 추론 프레임워크가 데이터 전송을 명시적으로 제어할 수 있어 운영 효율성을 높일 수 있다[ITME p.2]. FPGA 프로토타입을 통한 실험으로 ITME의 성능 잠재력을 저자 자체 평가로 검증하였다[ITME p.9][ITME p.11]. 반면, 장문맥 멀티턴 에이전트 서비스에서 중요한 병목인 대역폭과 지연, 전력 측면에서 ITME가 실제 운영 환경에서 충분히 해결하는지에 대한 구체적 수치는 부족하다[ITME p.2]. 상용 CXL 하드웨어의 제한과 PCIe Gen4 대비 Gen5의 대역폭 한계가 병목으로 작용할 가능성이 있다[ITME p.11]. 정확도 손실에 관한 ITME 고유의 데이터가 없으며, KV 캐시 오프로드가 문맥 집약적 작업에서 성능 저하를 유발할 수 있다는 보고가 있으나 ITME에 직접 적용된 근거는 없다[웹: KV Cache Offloading for Context-Intensive Tasks]. 운영 복잡도 증가에 대한 구체적 언급이 없어, 신규 인프라 도입이나 패치 관리 부담이 어느 정도인지 알 수 없다[ITME p.2]. 제3자 검증이나 실제 장문맥 멀티턴 에이전트 서비스 환경에서의 실증 데이터가 부족하다[ITME p.11].

MLA는 KV cache 크기를 MHA 대비 대폭 줄여 메모리 및 대역폭 병목을 효과적으로 완화한다[MLA p.6][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. 128K 토큰 문맥과 60개 층 모델에서 MLA는 7.5GB KV cache만 필요해 고성능 GPU에서 다중 동시 사용자 지원이 가능하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. MLA는 87.5% KV cache 압축 시 3% 정확도 손실로 장문맥 과제에서 허용 가능한 정확도 범위 내에 있다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any]. vLLM과 SGLang에서 MLA를 네이티브 지원해 별도 인프라 변경 없이 운영 복잡도를 낮춘다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. MLA 성능과 효율성은 DeepSeek-AI 공식 문서와 저자 보고, 일부 제3자 검증을 통해 실제 장문맥 멀티턴 에이전트 워크로드에서 측정되었다[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in]. 반면, MLA가 KV cache를 줄이면서도 MHA보다 성능이 우수하다고 하나, GQA와 MQA 대비 성능 비교는 부록에만 있어 상세 수치는 제한적이다[MLA p.6]. H200 GPU에서 KV cache FP4 경로가 없어 소프트웨어 에뮬레이션으로 처리되어 처리량 저하가 발생할 수 있다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. MLA의 정확도 손실 수치는 7B Llama2 모델과 LongBench 과제에 한정되어, DeepSeek-V2 자체 모델 정확도 손실 수치는 구체적으로 제시되지 않았다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any][MLA p.6].

## 5. 시사점

두 기술은 진입 비용의 방향이 정반대다. ITME는 기존 모델과 소프트웨어 스택을 그대로 두고, 신규 하드웨어(PCIe Gen5 기반 CXL-hybrid 메모리) 도입이 필요하다[ITME p.1][ITME p.11]. MLA는 기존 하드웨어를 그대로 두지만, 기존 MHA 또는 GQA 기반 LLM을 MLA로 바꾸려면 재학습이 필요하다[웹: Towards Economical Inference: Enabling DeepSeek's]. 이 대칭성은 도입을 검토하는 조직의 상황에 따라 평가가 달라질 수 있음을 보여준다.

시장성 관점에서는 ITME의 신규 하드웨어 도입이 도입 장벽으로 작용하며, 상용화 성숙도와 생태계 지지 근거가 부족하다[ITME p.11]. MLA는 기존 모델의 재학습 필요성과 범용 프레임워크 미지원이 도입 장벽으로 작용한다[웹: Predictive Multi-Tier Memory Management for KV Cache]. 두 기술 모두 시장 규모 수치는 상위 범주에 대한 것이고, 개별 기술의 시장성 수치는 없다[웹: CXL Memory Expansion Market Size & Share Report][웹: LLM Market Size to Hit USD 95.90 Billion by 2034 /].

도메인 관점에서는 실제 서비스 환경에서의 적용성과 검증 데이터가 중요하다. ITME는 장문맥 멀티턴 에이전트 서비스에서 대용량 메모리 병목을 해소할 수 있으나, 상용 CXL 하드웨어의 제한과 실제 운영 환경에서의 대역폭, 지연, 전력 측면 수치가 부족하다[ITME p.2][ITME p.11]. MLA는 KV cache를 대폭 줄여 메모리 병목을 완화하지만, 정확도 손실 수치는 일부 모델과 과제에 한정되어 있다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any][MLA p.6].

도메인을 바꿀 때 두 기술의 한계가 다르게 드러난다. ITME는 온디바이스 추론(스마트폰, 엣지 등) 환경에서는 PCIe Gen5 기반 CXL-hybrid 메모리 등 고대역폭, 대용량 메모리 인프라 전제가 맞지 않아 적용이 어렵다[ITME p.11][ITME p.2]. 클라우드 대규모 동시 서빙에서는 TB급 메모리 확장과 PCIe Gen5 대역폭을 활용할 수 있으나, 실제 클라우드 환경에서의 대역폭, 지연, 운영 복잡도에 대한 실증 데이터가 부족하다[ITME p.11][ITME p.2]. MLA는 온디바이스 환경에서 KV cache 크기를 줄여 유리할 수 있으나, MLA 적용을 위해 기존 모델의 재학습이 필요하고, 범용 프레임워크 미지원 시 도입 장벽이 존재한다[웹: Predictive Multi-Tier Memory Management for KV Cache][웹: Towards Economical Inference: Enabling DeepSeek's]. 클라우드 대규모 서빙에서는 KV cache를 줄여 다중 동시 세션 지원에 유리하지만, DeepSeek 계열 모델에 한정된 적용성과 기존 모델의 재학습 필요성이 확산의 제약이 될 수 있다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Towards Economical Inference: Enabling DeepSeek's].

관점이 달라지면 같은 기술도 평가가 달라진다. ITME는 시장성 관점에서는 도입 장벽이, 도메인 관점에서는 실제 적용성 검증 부족이 쟁점이 된다. MLA는 시장성 관점에서는 재학습 필요성과 생태계 확산성, 도메인 관점에서는 실제 워크로드에서의 성능과 정확도 손실이 주요 쟁점이다.

## 6. 한계점

이 평가는 1인 과제 범위로 한정했다. 네 관점 중 기술 성숙도(TRL) 관점과 이해관계자 관점은 범위에서 뺐다. 빠뜨린 것이 아니라 뺀 것이다.

두 기술의 공개 시점이 2024년 6월과 2026년 6월로 2년 차이 난다. 채택 현황을 나란히 놓으면 늦게 나온 쪽이 불리하게 읽힌다. 이 시간 차이를 감안하고 읽어야 한다.

시장 근거는 공개 자료만 썼다. 반도체 업계는 수율과 원가를 공개하지 않으므로 실제 채택 규모는 공개 정보로 확인할 수 없다. 웹 근거에 출처 등급을 붙였다. 일차는 논문과 코드 저장소와 벤더 공식 문서, 시장조사는 상업 조사기관 요약, 이차는 그 밖의 기사와 블로그다. 시장 규모 수치는 대부분 시장조사 등급이며 원본 보고서를 열어 대조하지 못했다. 같은 범주를 두고도 조사기관마다 수치가 크게 다른 경우가 있으므로 단일 수치를 그대로 믿으면 안 된다.

수집된 웹 근거의 등급 분포는 다음과 같다. ITME: 일차 14건, 이차 42건, 시장조사 0건. MLA: 일차 7건, 이차 49건, 시장조사 0건.

확증편향 감사에서 한쪽으로 기운 항목은 나오지 않았다. 취한 조치는 다음과 같다. 대칭_질의: 두 기술에 같은 질문 템플릿을 쓴다. 기술 이름만 바꾼다. 찬반_대칭: 관점마다 유리한 근거와 불리한 근거를 각각 따로 검색한다. 한쪽만 모이면 감사 단계에서 재수집한다. 순서_효과: 기술 제시 순서를 바꿔 두 번 실행하고, 결론이 달라지는지 기록한다. 근거_강제: 모든 평가 문장은 근거 원장의 id를 달아야 한다. 못 다는 문장은 버린다. 수치_대조: 보고서에 등장하는 숫자가 근거 원문에 있는지 코드로 대조한다. LLM 채점기만으로는 놓친다. 우열_금지: 한쪽을 높이거나 권하는 표현을 금지어 목록으로 두고 검사한다.

웹 근거가 개별 기술이 아니라 범주 수준이었던 항목이 있다. ITME의 시장 규모, MLA의 시장 규모, ITME의 일부 도메인 적용 근거는 CXL 메모리 확장 또는 LLM 전체 범주에 대한 것이며, 개별 기술 고유의 수치는 아니다. 이 점을 감안해야 한다.

## REFERENCE

본문에 인용 꼬리표로 실제 사용한 자료만 적는다. 목록은 근거 원장에서 자동으로 뽑았다.

**논문**

- DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434. https://arxiv.org/abs/2405.04434
- Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556. https://arxiv.org/abs/2606.12556

**웹 자료**

- Pyimagesearch(작성일 미상). *Build DeepSeek-V3: Multi-Head Latent Attention (MLA)*. pyimagesearch.com, https://pyimagesearch.com/2026/03/16/build-deepseek-v3-multi-head-latent-attention-mla-architecture
- Snsinsider(작성일 미상). *CXL Memory Expansion Market Size & Share Report*. snsinsider.com, https://www.snsinsider.com/reports/cxl-memory-expansion-market-10881
- Substack(작성일 미상). *Decoding Multi-Head Latent Attention (Part 1): The KV*. vizuara.substack.com, https://vizuara.substack.com/p/decoding-multi-head-latent-attention
- arXiv(작성일 미상). *DualPath: Breaking the Storage Bandwidth Bottleneck in*. arxiv.org, https://arxiv.org/html/2602.21548v1
- ACL Anthology(작성일 미상). *Enabling DeepSeek's Multi-Head Latent Attention in Any*. aclanthology.org, https://aclanthology.org/2025.acl-long.1597.pdf
- arXiv(작성일 미상). *KV Cache Offloading for Context-Intensive Tasks*. arxiv.org, https://arxiv.org/html/2604.08426v1
- Trendxinsights(작성일 미상). *LLM Market Size to Hit USD 95.90 Billion by 2034 /*. trendxinsights.com, https://trendxinsights.com/syndicated-market-research-reports/llm-market
- Spheron Network(작성일 미상). *Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV*. spheron.network, https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud
- arXiv(작성일 미상). *Predictive Multi-Tier Memory Management for KV Cache*. arxiv.org, https://arxiv.org/html/2604.26968v2
- Alphaxiv(작성일 미상). *Towards Economical Inference: Enabling DeepSeek's*. alphaxiv.org, https://www.alphaxiv.org/abs/2502.14837
