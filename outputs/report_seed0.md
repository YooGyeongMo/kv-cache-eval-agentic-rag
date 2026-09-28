# SUMMARY

이 보고서는 장문맥 멀티턴 에이전트 서비스에서 발생하는 KV cache 병목을 해결하기 위한 두 가지 대표적 접근, DeepSeek-V2 Multi-head Latent Attention(MLA)과 ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories(ITME)를 비교한다. MLA는 소프트웨어적으로 어텐션 구조를 바꿔 KV cache 데이터를 작게 만들고, ITME는 하드웨어적으로 CXL-hybrid 메모리 계층을 도입해 담을 공간을 넓힌다. 두 기술은 진입 비용, 도입 경로, 생태계 지지, 운영 효율, 도메인 적용성 등에서 관점에 따라 평가가 갈린다. 시장성 관점에서는 추론 스택 통합, 하드웨어 요구, 생태계 표준화 여부가, 도메인 적용 관점에서는 실제 운영 효율, 정확도 허용치, 하드웨어 가용성, 도입 장벽이 쟁점이 된다. 동일 기술도 적용 환경에 따라 평가가 달라지며, 두 기술은 진입 비용의 방향이 정반대라는 점에서 대조적이다.

## 1. 분석 배경

장문맥 멀티턴 에이전트 서비스에서는 단일 세션의 문맥이 100K 토큰을 넘고, 대화가 이어지면서 KV cache가 턴마다 누적된다. 이때 KV cache는 메모리 용량과 대역폭, 지연, 전력 측면에서 병목이 된다. 기존 Transformer 기반 LLM은 각 토큰마다 Key와 Value를 저장하는데, 문맥이 길어질수록 저장해야 할 KV cache가 기하급수적으로 늘어난다. 멀티턴 대화에서는 이전 턴의 KV cache가 계속 누적되어, 세션이 길어질수록 메모리 사용량이 급증한다. 이로 인해 GPU 메모리 한계에 부딪히거나, 대규모 동시 세션에서 처리량이 급격히 떨어진다.

이 병목을 해결하는 접근은 크게 두 가지로 나뉜다. 첫째는 소프트웨어적으로 KV cache 데이터를 작게 만드는 방법이다. 어텐션 구조를 바꿔 처음부터 KV cache가 작게 나오도록 하거나, 사후 압축을 통해 이미 생성된 KV cache를 줄인다. 둘째는 하드웨어적으로 담을 공간을 넓히는 방법이다. GPU나 호스트 메모리 외에 새로운 메모리 계층(CXL-hybrid 등)을 도입해 대용량 KV cache를 오프로드한다. 이 두 접근은 병목의 원인(데이터 크기 vs 저장 공간)에 대한 해석과, 실제 도입 경로에서 요구하는 변화가 정반대다.

소프트웨어 접근은 기존 하드웨어를 그대로 두고 모델 구조를 바꿔야 한다. 반면 하드웨어 접근은 모델을 그대로 두고 새로운 메모리 하드웨어를 들여야 한다. 이처럼 같은 병목을 두고 무엇을 바꿀 것인지가 정반대이기 때문에, 관점에 따라 평가가 달라질 수밖에 없다. 예를 들어, 소프트웨어 접근은 모델 재학습이나 미세조정이 필요해 도입 비용이 높아질 수 있고, 하드웨어 접근은 신규 인프라 도입과 운영 복잡도가 부담이 될 수 있다. 또한, 각 접근이 실제로 해결하는 병목의 범위, 적용 가능한 도메인, 생태계에서의 지지 정도도 다르다.

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

### MLA (DeepSeek-V2 Multi-head Latent Attention)

MLA는 Transformer 기반 LLM의 추론에서 발생하는 KV cache 병목을 해결하기 위해 어텐션 구조 자체를 바꾼다. 기존 Multi-Head Attention(MHA)에서는 각 토큰마다 Key와 Value를 저장하지만, MLA는 Key와 Value를 저차원(latent) 벡터로 결합해 KV cache를 공동 압축한다. 이 과정은 어텐션 모듈 내부에서 이루어지며, 별도의 사후 압축 단계 없이 처음부터 작은 KV cache가 생성된다[MLA p.1].

MLA의 동작은 다음과 같다. 입력 토큰이 들어오면, 각 어텐션 헤드에서 Key와 Value를 생성하는 대신, 이 둘을 저차원 공간에서 결합해 하나의 latent vector로 만든다. 이 latent vector는 기존 KV cache보다 훨씬 작은 크기를 가지며, 어텐션 계산 시에도 이 latent vector만을 사용한다. 이로써 토큰당 저장해야 할 KV cache의 양이 크게 줄어든다. MLA는 Transformer 프레임워크에서 어텐션 모듈이 KV cache를 사용하는 구조를 전제로 하며, low-rank key-value joint compression이 가능한 경우에만 적용할 수 있다.

보고된 수치는 MLA가 기존 MHA 대비 KV cache 요구량을 크게 줄인다는 것이다. MLA는 93.3퍼센트의 압축률을 기록했다[MLA p.1]. 128 attention heads, head_dim=128, d_c=512 설정에서 토큰당 1,024바이트의 KV cache가 필요하며, 128K 문맥, 60개 층 기준으로 7.5GB KV cache가 필요하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV].  측정은 LongBench 장문맥 과제에서 BF16 정밀도로 수행되었다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any].

MLA의 한계에 대해 저자가 직접 밝힌 내용은 없다. 다만, MLA는 기존 MHA 또는 GQA 기반의 잘 훈련된 LLM과 아키텍처가 달라, 도입 시 MHA2MLA와 같은 데이터 효율적 미세조정 프레임워크를 통한 재학습이 필요하다[웹: Towards Economical Inference: Enabling DeepSeek's]. 또한, MLA를 지원하는 범용 추론 프레임워크는 아직 없으며, MLA 모델은 MHA와 동등한 메모리 크기를 요구하는 비효율이 발생할 수 있다[웹: Predictive Multi-Tier Memory Management for KV Cache].

### ITME (Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories)

ITME는 대규모 LLM 추론에서 발생하는 KV cache 병목을 하드웨어 계층 확장으로 해결한다. ITME는 데이터 타입의 예측 가능성과 성능 민감도에 따라 LLM 데이터를 여러 계층으로 분리한다. 활성화와 작업 KV cache는 고속 GPU 또는 호스트 메모리 계층에 두고, 대용량이면서 예측 가능한 모델 가중치와 장문맥 KV cache는 원격 CXL-hybrid 메모리 확장 계층에 오프로드한다[ITME p.2].

ITME의 동작은 다음과 같다. 우선, LLM 추론 시 필요한 데이터(모델 가중치, 활성화, KV cache 등)를 예측 가능성과 성능 민감도에 따라 분류한다. 활성화와 작업 KV cache는 지연에 민감하므로 GPU 또는 호스트 메모리에 유지한다. 반면, 대용량이면서 예측 가능한 모델 가중치와 장문맥 KV cache는 PCIe Gen5 기반의 CXL-hybrid 메모리 계층에 저장한다. ITME는 하드웨어와 소프트웨어가 협조적으로 프리페칭을 수행해, NVMe SSD에서 CXL-hybrid 메모리의 내부 DRAM 캐시로 데이터를 미리 옮긴다. 그 결과 저장소 및 네트워크 지연을 숨기고, TB 단위의 KV cache 워크로드를 효율적으로 처리한다[ITME p.2][ITME p.11].

보고된 수치는 ITME가 35.7퍼센트의 처리량 개선을 보였다는 것이다.[ITME p.10] 이 수치는 CPU-offload 베이스라인(ShareGPT 데이터셋, 35턴, 256 동시 대화, recomputation 베이스라인 정규화) 대비 ITME 적용 시의 throughput 개선을 나타낸다. CPU-offload 베이스라인에서는 128GB 호스트 메모리가 캐싱에 할당되었고, ITME는 이보다 적은 호스트 메모리를 staging buffer로 사용했다[ITME p.11].

ITME의 성립 조건은 다음과 같다. PCIe Gen5 인터페이스를 사용하는 CXL-hybrid 메모리 계층이 필요하며, 대규모 LLM 워크로드와 TB 단위 KV cache 요구, 예측 가능한 데이터 타입(모델 가중치, 장문맥 KV cache)이 전제된다. 하드웨어-소프트웨어 협조적 프리페칭 전략과 NVMe SSD 연동도 필요하다.

저자가 밝힌 한계는, ITME가 강한 경쟁 상황에서 예측 불가능한 I/O stall로 인해 성능 변동이 발생할 수 있다는 점이다. 이는 read-priority 스케줄링으로 완화된다고 한다. 그 외의 명시적 한계는 근거에 없다[ITME p.11].

## 4. 관점별 평가

### 4.1 시장성 관점

| 기준        | MLA | ITME |
|-------------|-----|------|
| 채택 폭 | DeepSeek-V2의 MLA는 vLLM, SGLang, TensorRT-LLM 등 주요 추론 스택에서 커널 및 서빙 엔진 수준으로 구현되어 있다. vLLM은 config.json에서 MLA를 자동 인식하고, SGLang은 FlashInfer MLA 커널을 통해 지원한다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Predictive Multi-Tier Memory Management for KV Cache]. | ITME는 FlexGen, DeepSpeed Inference, LMCache 등 기존 소프트웨어 스택과 호환되도록 설계되었으나, vLLM, SGLang, TensorRT-LLM, llama.cpp, HuggingFace transformers 등 주요 추론 스택이나 상용 제품의 커널, 서빙 엔진, SDK 계층에 구현되었다는 직접적 근거는 없다[ITME p.11][ITME p.1]. |
| 채택 깊이 | vLLM에서는 DeepSeek 계열 모델에 대해 MLA가 기본값으로 자동 처리되며, SGLang에서는 특정 플래그를 통해 MLA 커널을 활성화할 수 있다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | ITME 문서에는 구현이 기본값인지, 선택 옵션인지, 실험 기능인지에 대한 안정성 등급이나 기본값 여부가 명시되어 있지 않다[ITME p.11]. |
| 진입 비용 | 기존 MHA 또는 GQA 기반 LLM은 MLA 아키텍처와 호환되지 않아, MLA를 도입하려면 MHA2MLA 미세조정 프레임워크를 통해 재학습이 필요하다. 새 하드웨어 구매에 대한 직접 언급은 없으나, H200, B200 등 특정 GPU에서 최적화 플래그가 존재한다[웹: Towards Economical Inference: Enabling DeepSeek's][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | ITME는 PCIe Gen5 기반 CXL-hybrid 메모리 등 신규 하드웨어 도입이 요구된다. 기존 가중치 재학습 필요성에 대한 언급은 없다[ITME p.2][ITME p.11]. |
| 생태계 지지 | MLA는 DeepSeek의 핵심 혁신으로, MHA2MLA 미세조정 프레임워크가 발표되어 기존 모델의 전환을 지원한다. 표준 규격, 컨소시엄, 독립 제3자 재현 결과에 대한 언급은 없다[웹: Towards Economical Inference: Enabling DeepSeek's]. | ITME는 CXL 기반 메모리 확장 연구의 일환이나, ITME 자체가 표준 규격이나 컨소시엄에 포함되었다는 근거는 없다. 후속 연구가 ITME를 베이스라인으로 삼았다는 독립 재현 결과도 없다[ITME p.11]. |
| 시장 수치 | LLM 시장 규모가 2025년 6.48 billion USD에서 2034년 95.90 billion USD로 성장할 것으로 TrendX Insights Research가 발표했다. 이 수치는 LLM 상위 범주를 가리키며, MLA 기술 자체를 직접 지칭하지는 않는다[웹: LLM Market Size to Hit USD 95.90 Billion by 2034 /]. | CXL 메모리 확장 시장은 2026년에 1.74 billion 달러 규모로 성장할 전망이다. 이 수치는 CXL 메모리 확장 상위 범주를 가리키며, ITME 기술 자체를 직접 지칭하지는 않는다[웹: CXL Memory Expansion Market Size & Share Report]. |

### 4.2 도메인 적용 관점

| 기준        | MLA | ITME |
|-------------|-----|------|
| 병목 일치 | 장문맥 멀티턴 에이전트 서비스에서 KV cache가 메모리 용량과 대역폭 병목의 핵심이며, MLA는 KV cache 크기를 크게 줄여 이 병목과 일치하는 해결책을 제시한다[MLA p.6][웹: DualPath: Breaking the Storage Bandwidth Bottleneck in][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | 장문맥 멀티턴 에이전트 서비스에서 병목은 메모리 용량과 대역폭, 지연, 전력이며, ITME는 대용량 모델 가중치와 장문맥 KV cache를 원격 확장 계층에 오프로드하여 메모리 용량 병목을 해결한다. 활성화 및 작업 KV cache는 지연에 민감해 ITME가 직접 다루지 않는다[ITME p.2][웹: HyMCache: A KV Cache Frameworkfor Multi-Turn LLM][웹: Using CXL Fabric-Attached Memory to Enable Shared KV]. |
| 자원 전제 | MLA는 128 attention heads, head_dim=128, d_c=512 설정에서 토큰당 1,024바이트 KV cache를 요구하며, 128K 문맥, 60개 층 기준으로 7.5GB KV cache가 필요하다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Build DeepSeek-V3: Multi-Head Latent Attention (MLA)]. | ITME는 PCIe Gen5 인터페이스를 사용하는 CXL-hybrid 메모리 아키텍처를 전제로 하며, 하드웨어 수준의 프리페칭과 사용자 수준 API를 요구한다[ITME p.2][ITME p.11][웹: CXL 4.0 Infrastructure Planning Guide / Introl Blog]. |
| 정확도 허용치 | 근거 없음  측정은 LongBench 장문맥 과제에서 BF16 정밀도로 수행되었다[웹: Enabling DeepSeek's Multi-Head Latent Attention in Any]. | ITME의 정확도 손실 수치나 측정 조건이 근거에 없다. 다른 연구에서 KV 캐시 오프로드 시 context-intensive 작업에서 성능 저하가 보고되었다[웹: KV Cache Offloading for Context-Intensive Tasks][ITME p.9]. |
| 운영 비용 | MLA는 vLLM과 SGLang에서 네이티브 지원되어 별도 플래그 없이 활성화 가능하며, 추가 전송 계층이나 패치 관리 없이 기존 GPU 클라우드 인프라에서 운영 가능하다. H200 GPU에서 KV cache FP4 저장은 소프트웨어 에뮬레이션으로 처리되어 처리량 저하가 발생할 수 있다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | ITME는 하드웨어-소프트웨어 협조적 프리페칭과 사용자 수준 API를 제공하지만, 별도의 패치 관리나 신규 인프라 도입, 전송 계층 추가에 대한 구체적 언급은 없다. CXL 메모리 도입 시 OS 커널, 메모리 할당기, AI 프레임워크의 지원이 필요하다[ITME p.2][웹: CXL 4.0 Infrastructure Planning Guide / Introl Blog]. |
| 증거 성숙도 | MLA 성능과 효율성 수치는 DeepSeek 저자가 직접 보고한 실험 결과이며, LongBench 벤치마크와 실제 GPU 클라우드 환경에서 측정되었으나 제3자 검증 근거는 없다[MLA p.6][웹: Enabling DeepSeek's Multi-Head Latent Attention in Any][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | ITME는 SK hynix CMM과 Gen5 PCIe 기반 FPGA 프로토타입을 사용해 성능 잠재력을 검증했으나, 상용 CXL 하드웨어가 제한적이고 일부 연구는 시뮬레이션이나 에뮬레이션에 의존한다. 도메인 내 실제 대규모 배포 사례는 부족하다[ITME p.11][ITME p.9][웹: Scalable Processing-Near-Memory for 1M-Token LLM]. |

## 5. 시사점

### 진입 비용의 정반대 방향

MLA와 ITME는 진입 비용의 방향이 정반대다. MLA는 기존 하드웨어를 그대로 두고 모델 구조를 바꿔야 하며, 기존 MHA 또는 GQA 기반 LLM을 MLA로 전환하려면 미세조정이나 재학습이 필요하다[웹: Towards Economical Inference: Enabling DeepSeek's]. 반면 ITME는 모델을 그대로 두고 PCIe Gen5 기반 CXL-hybrid 메모리 등 신규 하드웨어를 도입해야 한다[ITME p.2][ITME p.11]. 이로 인해, 한쪽은 소프트웨어적 진입 장벽이, 다른 쪽은 하드웨어적 진입 장벽이 된다.

### 관점 간 상충

관점별로 같은 사실이 다르게 읽힌다. MLA의 경우, 시장성 관점에서는 vLLM, SGLang, TensorRT-LLM 등 주요 추론 스택에서 커널 및 서빙 엔진 수준으로 구현되어 있어 실용적이고, vLLM에서는 MLA가 기본값으로 자동 활성화되어 사용 편의성이 높다고 평가된다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV][웹: Predictive Multi-Tier Memory Management for KV Cache]. 도메인 적용 관점에서는 실제 운영 시 하드웨어별 처리량 저하(예: FP4 소프트웨어 에뮬레이션) 등 세부적인 인프라 제약이 드러난다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. 시장성 관점은 '지원 여부'에, 도메인 관점은 '실제 운영 효율'에 더 무게를 둔다.

생태계 및 검증 성숙도에서도 차이가 있다. 시장성 관점에서는 표준 규격, 컨소시엄, 독립 제3자 재현 결과가 공개되어 있지 않아 생태계 지지가 제한적이라고 본다[웹: Towards Economical Inference: Enabling DeepSeek's]. 도메인 적용 관점에서는 성능 수치가 저자 자체 보고에만 의존한다는 점을 지적한다[MLA p.6][웹: Enabling DeepSeek's Multi-Head Latent Attention in Any][웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. 시장성은 '공식적 지지'에, 도메인은 '외부 검증'에 초점을 둔다.

ITME의 경우, 시장성 관점에서는 기존 다중 계층 KV 캐시 및 프리페칭 소프트웨어 스택과 호환되도록 설계되어 기존 인프라와의 연계 가능성을 높인다고 평가한다[ITME p.11]. 그러나 ITME가 주요 LLM 추론 스택이나 상용 제품의 커널, 서빙 엔진, SDK 계층에 구현되었다는 직접적 근거가 없고, 신규 하드웨어가 필요하다는 점을 지적한다[ITME p.11][ITME p.1][ITME p.2]. 도메인 적용 관점에서는 실제 하드웨어 가용성과 운영 환경에서의 제약(상용 CXL 하드웨어의 제한, API 요구 등)에 더 초점을 둔다[ITME p.2][ITME p.11][웹: CXL 4.0 Infrastructure Planning Guide / Introl Blog]. 시장성은 '채택 가능성'에, 도메인은 '실제 적용 가능성'에 더 민감하다.

정확도 및 성능 검증에서도 상충이 있다. 시장성 관점에서는 ITME 자체의 정확도 손실 수치와 측정 조건이 근거에 없어 판단 불가로 처리한다[ITME p.11]. 도메인 관점에서는 유사 범주 연구의 성능 저하 사례까지 언급하며 실제 적용 시 위험성을 더 강조한다[웹: KV Cache Offloading for Context-Intensive Tasks][ITME p.9]. 시장성은 '공식 수치의 부재'에 머무르고, 도메인은 '실제 도메인에서의 잠재적 문제'까지 고려한다.

### 도메인 전환

| 도메인 | MLA가 막히는 지점 | ITME가 막히는 지점 |
|--------|------------------|-------------------|
| 장문맥 멀티턴 에이전트 서비스 | 없음 | 없음 |
| 온디바이스 추론 | 128K 문맥, 60개 층 기준 7.5GB KV cache 요구가 온디바이스 환경(스마트폰, 엣지 등)에서는 과도하여 적용이 어렵다. 온디바이스 적용에 대한 직접적 수치나 사례는 없다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. 근거 없음 | PCIe Gen5 기반 CXL-hybrid 메모리 등 대규모 외장 메모리 인프라를 전제로 하므로 온디바이스 환경에서는 적용이 불가능하다. 온디바이스 환경에는 CXL 인터페이스나 대용량 외장 메모리 확장이 현실적으로 제공되지 않는다[ITME p.2][ITME p.11]. |
| 클라우드 대규모 동시 서빙 | H200 GPU에서 FP4 저장이 소프트웨어 에뮬레이션으로 처리되어 처리량 저하가 발생할 수 있다[웹: Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV]. | 상용 CXL 하드웨어의 제한적 가용성과 OS/프레임워크 지원 필요성, 실제 대규모 배포 사례의 부재가 도입 장벽으로 작용한다[ITME p.11][웹: CXL 4.0 Infrastructure Planning Guide / Introl Blog]. |

이 표는 동일 기술도 적용 환경에 따라 막히는 지점이 다름을 보여준다. 온디바이스 추론에서는 MLA도 메모리 요구량이 크고, ITME는 하드웨어 인프라 자체가 불가능하다. 클라우드 대규모 동시 서빙에서는 MLA는 하드웨어별 처리량 저하, ITME는 하드웨어 가용성과 생태계 지원의 한계가 각각 걸림돌이 된다.

## 6. 한계점

이 평가는 1인 과제 범위로 한정했다. 네 관점 중 기술 성숙도(TRL) 관점과 이해관계자 관점은 범위에서 뺐다. 빠뜨린 것이 아니라 뺀 것이다. 두 기술의 공개 시점이 2024년 6월과 2026년 6월로 2년 차이 난다. 채택 현황을 나란히 놓으면 늦게 나온 쪽이 불리하게 읽힌다. 이 시간 차이를 감안하고 읽어야 한다.

시장 근거는 공개 자료만 썼다. 반도체 업계는 수율과 원가를 공개하지 않으므로 실제 채택 규모는 공개 정보로 확인할 수 없다. 웹 근거에 출처 등급을 붙였다. 일차는 논문, 코드 저장소, 벤더 공식 문서다. 시장조사는 상업 조사기관의 요약이고, 원본 보고서를 확인할 수 없다. 이차는 그 밖의 블로그와 기사다. 시장 규모 수치는 대부분 시장조사 등급이며, 조사기관마다 수치가 크게 다를 수 있으므로 단일 수치를 그대로 믿으면 안 된다.

수집된 웹 근거의 등급 분포는 MLA: 일차 7건, 이차 49건, 시장조사 0건. ITME: 일차 14건, 이차 42건, 시장조사 0건이다. 확증편향 감사에서 한쪽으로 기운 항목은 나오지 않았다. 취한 조치는 다음과 같다. 대칭 질의: 두 기술에 같은 질문 템플릿을 쓴다. 기술 이름만 바꾼다. 찬반 대칭: 관점마다 유리한 근거와 불리한 근거를 각각 따로 검색한다. 한쪽만 모이면 감사 단계에서 재수집한다. 순서 효과: 기술 제시 순서를 바꿔 두 번 실행하고, 결론이 달라지는지 기록한다. 근거 강제: 모든 평가 문장은 근거 원장의 id를 달아야 한다. 못 다는 문장은 버린다. 수치 대조: 보고서에 등장하는 숫자가 근거 원문에 있는지 코드로 대조한다. 우열 금지: 한쪽을 높이거나 권하는 표현을 금지어 목록으로 두고 검사한다.

웹 근거가 개별 기술이 아니라 범주 수준이었던 항목이 있다. MLA와 ITME 모두 시장 규모 수치는 각각 LLM 상위 범주, CXL 메모리 확장 상위 범주에 대한 전망일 뿐, 기술 자체의 시장 규모나 성장 전망 수치는 없다. 이 점은 해석에 주의가 필요하다.

> 이 보고서는 생성 후 검증 단계를 거쳤다. 인용 꼬리표가 가리키는 근거에 해당 내용이 없는 문장은 그 내용이 실제로 나온 출처로 꼬리표를 바로잡았고, 문서 풀 어디에도 근거가 없는 항목은 근거 없음으로 내렸다. 참고문헌의 링크는 접속을 확인했고 실패한 것은 그 사실을 적었다.

## REFERENCE

본문에 인용 꼬리표로 실제 사용한 자료만 적는다. 목록은 근거 원장에서 자동으로 뽑았다.

**논문**

- DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv*, 2405.04434. https://arxiv.org/abs/2405.04434
- Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J.(2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. *arXiv*, 2606.12556. https://arxiv.org/abs/2606.12556

**웹 자료**

- Pyimagesearch(작성일 미상). *Build DeepSeek-V3: Multi-Head Latent Attention (MLA)*. pyimagesearch.com, https://pyimagesearch.com/2026/03/16/build-deepseek-v3-multi-head-latent-attention-mla-architecture
- Introl(작성일 미상). *CXL 4.0 Infrastructure Planning Guide / Introl Blog*. introl.com, https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-memory-pooling-2025
- Snsinsider(작성일 미상). *CXL Memory Expansion Market Size & Share Report*. snsinsider.com, https://www.snsinsider.com/reports/cxl-memory-expansion-market-10881
- arXiv(작성일 미상). *DualPath: Breaking the Storage Bandwidth Bottleneck in*. arxiv.org, https://arxiv.org/html/2602.21548v1
- ACL Anthology(작성일 미상). *Enabling DeepSeek's Multi-Head Latent Attention in Any*. aclanthology.org, https://aclanthology.org/2025.acl-long.1597.pdf
- arXiv(작성일 미상). *HyMCache: A KV Cache Frameworkfor Multi-Turn LLM*. arxiv.org, https://arxiv.org/html/2607.18141v2
- arXiv(작성일 미상). *KV Cache Offloading for Context-Intensive Tasks*. arxiv.org, https://arxiv.org/html/2604.08426v1
- Trendxinsights(작성일 미상). *LLM Market Size to Hit USD 95.90 Billion by 2034 /*. trendxinsights.com, https://trendxinsights.com/syndicated-market-research-reports/llm-market
- Spheron Network(작성일 미상). *Multi-Head Latent Attention (MLA) on GPU Cloud: Cut KV*. spheron.network, https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud
- arXiv(작성일 미상). *Predictive Multi-Tier Memory Management for KV Cache*. arxiv.org, https://arxiv.org/html/2604.26968v2
- arXiv(작성일 미상). *Scalable Processing-Near-Memory for 1M-Token LLM*. arxiv.org, https://arxiv.org/html/2511.00321v1
- Alphaxiv(작성일 미상). *Towards Economical Inference: Enabling DeepSeek's*. alphaxiv.org, https://www.alphaxiv.org/abs/2502.14837
- SNIA(작성일 미상). *Using CXL Fabric-Attached Memory to Enable Shared KV*. snia.org, https://www.snia.org/sniadeveloper/session/19665 (접속 확인 실패, 2026-09-28 기준)
