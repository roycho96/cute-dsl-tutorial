# References

Last reviewed: 2026-08-31

각 장에서 사용한 documentation, source code, paper, technical blog를 기록합니다. API와 architecture의 동작은 NVIDIA documentation과 CUTLASS source를 우선합니다. 논문과 blog는 수학적 설명, kernel 구성, 구현 사례를 보완하는 데 사용합니다.

## NVIDIA CUTLASS and CuTe DSL

- [CUTLASS 4.6.1](https://github.com/NVIDIA/cutlass)
  - CuTe DSL API, examples, tests, release notes
- [CuTe DSL Quick Start Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
  - supported platform, Python and CUDA versions, installation
- [DSL Programming Model: Introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
  - `@cute.jit`, `@cute.kernel`, JIT arguments
- [End-to-End Code Generation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html)
  - AST rewrite, tracing, MLIR lowering, `preprocess` mode
- [Educational Notebooks](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/notebooks.html)
  - hello world, Tensor, Layout algebra, elementwise add
- [CuTe DSL API Reference](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html)
  - `Layout`, `Tensor`, partitioning, copy, MMA
- [MMA Programming Guides](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/index.html)
  - warp MMA, WGMMA, `tcgen05`
- [Warp-level Matrix Multiply-Accumulate Programming](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html)
  - `MmaF16BF16Op`, MMA atom, multi-warp `TiledMMA`, operand fragment
- [Debugging Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html)
  - IR, PTX, CUBIN, SASS, Compute Sanitizer

## CuTe Layout and Tensor

- [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
- [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)
- [CuTe Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html)
- [CuTe Algorithms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html)

Python DSL과 C++ CuTe는 syntax가 다르지만 `Layout`과 `Tensor`의 정의를 공유합니다. Part 1과 Part 2는 이 문서들의 정의와 현재 Python API를 함께 사용합니다.

## CUDA and PTX

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Samples: Reduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction)
- [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)

TMA, `mbarrier`, WGMMA, `tcgen05`, thread block cluster, memory ordering은 DSL wrapper뿐 아니라 underlying instruction의 contract도 확인합니다.

## NVIDIA technical articles and talks

- [Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)
- [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [CUTLASS Python DSL Infrastructure](https://llvm.org/devmtg/2025-10/slides/technical_talks/ozen.pdf)
- [DSLs for LLM Kernels](https://hc2025.hotchips.org/assets/program/tutorials/dsl_llm_kernels.pdf)

## FlashAttention

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
- [FlashAttention-4 CuTe DSL source](https://github.com/Dao-AILab/flash-attention/tree/29e40cfc420da5cbf2d97a1c273483e0f04b57b2/flash_attn/cute)

FA1부터 FA4까지 같은 attention 연산을 구현하되, 각 논문에서 바뀐 algorithm, work partitioning, asynchronous pipeline, architecture-specific dataflow를 구분해 적용합니다. CuTe DSL API와 kernel 구성은 현재 FA4 source를 기준으로 확인합니다.

## Colfax Research

- [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf)
- [Writing GEMM Kernels Using Tensor Memory for NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)

Layout algebra, TMEM access, 1-SM and 2-SM UMMA, scale-factor Layout을 설명할 때 사용합니다. API는 현재 CUTLASS source와 다시 대조합니다.

## Additional technical writing

### Simon Veitner

- [CuTe DSL articles](https://veitner.bearblog.dev/blog/)
- [An applied introduction to CuTeDSL](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/)
- [SGEMM in CuTeDSL](https://veitner.bearblog.dev/sgemm-in-cutedsl/)
- [CuTe partitions](https://veitner.bearblog.dev/cute-partitions/)
- [MMA Atoms in CuTe](https://veitner.bearblog.dev/mma-atoms-in-cute/)

### Layout tutorials and papers

- [CuTe DSL from scratch](https://bikrammajhi.github.io/blogs/cute-dsl-from-scratch/)
- [learn-cutedsl](https://github.com/luongthecong123/learn-cutedsl)
- [CuTe Layout Representation and Algebra](https://arxiv.org/abs/2603.02298)

## Sources by chapter

| Chapters | Primary sources |
|---|---|
| 01. Execution model | DSL Introduction, End-to-End Code Generation, official elementwise-add notebook |
| 02. Shape, Stride, and Layout | CuTe Layout, Python Core API, Layout algebra notebook, Colfax Layout algebra note |
| 03. Tensor, slicing, and tiling | CuTe Tensor and Algorithms, Python Core API, official Tensor notebooks, Simon Veitner |
| 04. Vector addition | official elementwise-add notebook, Python Core API, CUDA Programming Guide, CuTe DSL Debugging Guide |
| 05. Warp and block reduction | CUDA Programming Guide, CUDA reduction sample, CUTLASS `cta_norm.py`, `SmemAllocator` source |
| 06. Shared-memory transpose and swizzle | CUDA Programming Guide, NVIDIA transpose article, CuTe `Swizzle` source, CuTe DSL GEMM example |
| 07. Row-wise softmax | CUDA Programming Guide, CuTe math API, CUTLASS `cta_norm.py`, online normalizer paper |
| 08. MMA atom and TiledMMA | warp MMA programming guide, PTX `mma.sync`, CuTe `atom.py`, Veitner MMA atom article |
| 09. First Tensor Core GEMM | CUTLASS Ampere `tensorop_gemm.py`, CuTe GEMM tutorial, PTX `ldmatrix` |
| 10. Multistage GEMM and epilogue | CUTLASS Ampere `tensorop_gemm.py`, `cp.async` API and PTX, CuTe Tensor algorithms |
| 11~15. TMA and architecture | PTX ISA, Pipeline API, NVIDIA guides and notebooks |
| 16~17. Complete GEMM kernels | CUTLASS examples and tests, Colfax Research, Nsight documentation |
| 18~21. FlashAttention | FA1~FA4 papers, official FlashAttention-4 CuTe DSL source, CUTLASS attention examples |
| 22. Grouped GEMM and MoE | CUTLASS grouped GEMM examples, Nsight documentation, model-derived expert shapes |

## Figures

| Figure | Local file | Original | License |
|---|---|---|---|
| Figure 1-2 | `assets/nvidia-cute-dsl-compilation.png` | [NVIDIA CUTLASS `dsl_compilation.png`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pythonDSL/cute_dsl_general/dsl_compilation.png) | BSD-3-Clause, [notice](../THIRD_PARTY_NOTICES.md#nvidia-cutlass-documentation-figure) |

Third-party figure는 명시적인 재배포 조건이 있을 때만 저장소에 포함합니다. License가 확인되지 않은 figure는 원본 article을 reference로 남기고 필요한 diagram을 새로 작성합니다.
