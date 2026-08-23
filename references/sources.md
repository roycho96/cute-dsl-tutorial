# 출처 지도

마지막 확인: 2026-08-23

이 문서는 책을 집필할 때 확인할 자료와 사용 범위를 기록합니다. 기술 사실은 공식 문서와 source code를 우선하며, 기술 블로그는 설명 방식과 구현 사례를 교차 검증하는 데 사용합니다.

외부 자료는 단순히 링크만 모으지 않고 장의 정의, 설명 순서, 예제 설계에 반영합니다. 재배포 라이선스가 확인되는 그림은 저장소에 포함하고 캡션에 저작자·원문·라이선스를 표시합니다. 재배포 조건이 명확하지 않은 그림은 복제하지 않고, 필요한 개념을 새 도식으로 다시 구성한 뒤 참고한 글을 밝힙니다. 예제 코드는 현재 CuTe DSL API로 직접 작성하고 실행해 확인합니다.

## 1. 기준 자료

### NVIDIA CUTLASS와 CuTe DSL

- [CUTLASS 4.6.1 repository](https://github.com/NVIDIA/cutlass)
  - 현재 API 이름, examples, test, release note의 기준입니다.
  - `examples/python/CuTeDSL`과 `test/python/cute`를 문서보다 먼저 확인해야 하는 경우가 있습니다.
- [CuTe DSL Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
  - 지원 OS·Python·CUDA 조합과 설치 방법의 기준입니다.
- [DSL Programming Model: Introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
  - `@cute.jit`, `@cute.kernel`, 호출 관계와 compile model의 기준입니다.
- [End-to-End Code Generation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html)
  - AST rewrite, tracing, MLIR lowering, `preprocess` mode의 기준입니다.
- [Educational Notebooks](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/notebooks.html)
  - hello world, tensor, layout algebra, asynchronous pipeline의 최소 예제를 확인합니다.
- [CuTe DSL API Reference](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html)
  - `Layout`, `Tensor`, partition, copy, MMA API signature의 기준입니다.
- [MMA Programming Guides](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/index.html)
  - Ampere warp MMA, Hopper WGMMA, Blackwell `tcgen05`의 instruction 조건을 확인합니다.
- [Debugging Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html)
  - IR, PTX, CUBIN, SASS dump 방법의 기준입니다.

### CuTe의 수학적 기반

- [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
- [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
- [CuTe Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html)
- [CuTe Algorithms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html)

Python DSL과 C++ CuTe의 표면 문법은 다르지만 Layout과 Tensor의 의미는 같습니다. Layout 장에서는 위 네 문서를 정의의 기준으로 사용합니다.

### CUDA와 PTX

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

TMA, `mbarrier`, WGMMA, `tcgen05`, cluster, memory ordering을 설명할 때는 DSL wrapper가 아니라 underlying instruction의 계약까지 확인합니다.

## 2. NVIDIA 기술 글과 발표

- [Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)
  - CuTe DSL과 C++ CuTe의 대응 관계, TiledMMA의 위치, 공식 성능 측정 범위를 확인합니다.
- [CUTLASS Python DSL Infrastructure](https://llvm.org/devmtg/2025-10/slides/technical_talks/ozen.pdf)
  - DSL compiler pipeline과 MLIR 기반 code generation의 전체 구조를 확인합니다.
- [DSLs for LLM Kernels](https://hc2025.hotchips.org/assets/program/tutorials/dsl_llm_kernels.pdf)
  - CuTe DSL이 적합한 kernel 범위와 Blackwell GEMM 사례를 확인합니다.

## 3. 구현 해설과 교차 검증

### Colfax Research

- [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf)
- [Writing GEMM Kernels Using Tensor Memory for NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)

Layout algebra, TMEM load, 1-SM·2-SM UMMA, scale-factor Layout을 공식 example과 대조할 때 사용합니다.

### Simon Veitner

- [CuTe DSL 글 목록](https://veitner.bearblog.dev/blog/)
- [An applied introduction to CuTeDSL](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/)
- [SGEMM in CuTeDSL](https://veitner.bearblog.dev/sgemm-in-cutedsl/)
- [CuTe partitions](https://veitner.bearblog.dev/cute-partitions/)
- [MMA Atoms in CuTe](https://veitner.bearblog.dev/mma-atoms-in-cute/)

작은 kernel에서 partition과 TiledMMA로 넘어가는 설명 순서를 검토할 때 사용합니다. API는 글의 작성 시점보다 현재 CUTLASS source를 우선합니다.

### 최근 학습 자료

- [CuTe DSL from scratch](https://bikrammajhi.github.io/blogs/cute-dsl-from-scratch/)
- [learn-cutedsl](https://github.com/luongthecong123/learn-cutedsl)
- [CuTe Layout Representation and Algebra](https://arxiv.org/abs/2603.02298)

계층적 Layout을 시각화하는 방법과 GEMM 최적화 단계를 비교합니다. 개인 repository의 성능 수치는 동일한 환경에서 재측정하지 않는 한 책의 결과로 인용하지 않습니다.

## 4. 장별 적용

| 범위 | 우선 확인할 자료 |
|---|---|
| 01. 실행 경계 | DSL Introduction, Quick Start, official hello-world notebook |
| 02~06. Layout | CuTe Layout/Tensor 공식 문서, layout algebra paper, Colfax note |
| 07~12. Partition·Copy | official elementwise example, CuTe Algorithms, Simon Veitner |
| 13~16. MMA·GEMM | official MMA guides와 tutorial GEMM source |
| 17~21. TMA·Pipeline | PTX ISA, official pipeline API, NVIDIA notebooks |
| 22~27. Hopper·Blackwell | tuning guides, official Hopper/Blackwell examples, Colfax |
| 28~33. 실제 kernel | CUTLASS examples/tests, Nsight documentation, 직접 측정 |

## 5. 외부 그림 사용 기록

| 책의 위치 | 자산 | 원문 | 사용 조건 |
|---|---|---|---|
| 01. 실행 경계 | `nvidia-cute-dsl-compilation.png` | NVIDIA CUTLASS `dsl_compilation.png` | BSD-3-Clause, [고지](../THIRD_PARTY_NOTICES.md#nvidia-cutlass-documentation-figure) |

Colfax Research의 글과 PDF는 설명과 구현을 교차 검증하는 주요 자료입니다. 다만 현재 공개 페이지에는 별도의 재배포 허가가 명시돼 있지 않으므로, 해당 사이트의 그림은 저장소에 복제하지 않습니다. 추후 저작자가 허가하거나 명시적인 라이선스가 붙은 자산을 확인하면 이 표에 원문과 조건을 추가합니다.
