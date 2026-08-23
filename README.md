# CuTe DSL Tutorial

CUDA 코드를 작성해 본 독자를 위한 한국어 CuTe DSL 교재입니다. 간단한 elementwise kernel에서 시작해 `Layout`, `TiledCopy`, `TiledMMA`, TMA pipeline, Hopper WGMMA, Blackwell TMEM까지 단계별로 다룹니다.

API를 나열하는 대신 각 abstraction이 CUDA hardware와 어떻게 연결되는지 설명합니다. 모든 주요 장에는 실행 가능한 코드, tensor layout을 손으로 추적하는 예제, architecture diagram이 포함됩니다.

![From Python to a GPU kernel](assets/01-execution-model.svg)

## Prerequisites

- CUDA grid, block, thread, warp
- global memory, shared memory, register
- coalescing과 bank conflict
- `__syncthreads()`와 기본적인 synchronization
- matrix multiplication `C = A @ B`

CUTLASS C++ template이나 Tensor Core instruction을 미리 알 필요는 없습니다. 예제는 Linux, Python 3.12, CuTe DSL 4.6.1에서 검증합니다.

설치와 코드를 읽는 순서는 [00. 시작하기](book/00-reading-guide.md)에 정리했습니다.

## Contents

### Part 1. CuTe fundamentals

- [x] [01. CuTe DSL execution model](book/01-execution-model.md)
- [ ] 02. Shape and IntTuple
- [ ] 03. Layout: coordinate to offset
- [ ] 04. Hierarchical Layout and slicing
- [ ] 05. Layout algebra: coalesce and composition
- [ ] 06. Layout algebra: complement, divide, and tile

### Part 2. From Layout to kernel

- [ ] 07. Tensor: Engine and Layout
- [ ] 08. CTA tiling and `local_tile`
- [ ] 09. Thread-value Layout
- [ ] 10. TiledCopy and vectorized copy
- [ ] 11. Shared-memory Layout and swizzle
- [ ] 12. Predication

### Part 3. Building a GEMM

- [ ] 13. MMA atom and TiledMMA
- [ ] 14. From SIMT GEMM to Tensor Core GEMM
- [ ] 15. GMEM → SMEM → RMEM dataflow
- [ ] 16. GEMM epilogue

### Part 4. Asynchronous pipelines

- [ ] 17. TMA tensor and TensorMap descriptor
- [ ] 18. `mbarrier` and PipelineState
- [ ] 19. Multistage pipeline
- [ ] 20. Warp-specialized kernels
- [ ] 21. Persistent tile scheduler

### Part 5. Hopper and Blackwell

- [ ] 22. Hopper WGMMA
- [ ] 23. Blackwell TMEM and `tcgen05`
- [ ] 24. Blackwell 1-SM GEMM
- [ ] 25. CTA pair and 2-SM MMA
- [ ] 26. TMA multicast and thread block clusters
- [ ] 27. NVFP4 block-scaled GEMM

### Part 6. Production kernels

- [ ] 28. Fused epilogue
- [ ] 29. Grouped GEMM and MoE
- [ ] 30. PyTorch, DLPack, and AOT integration
- [ ] 31. Correctness and numerical error
- [ ] 32. Reading IR, PTX, and SASS
- [ ] 33. Profiling with Nsight Compute

## Running the examples

```bash
source ~/workspace/.venv_wsl/bin/activate
python examples/01_execution_model/vector_add.py
```

## References and figures

기술적인 설명은 현재 NVIDIA documentation과 CUTLASS source를 기준으로 작성합니다. Colfax Research를 비롯한 논문과 technical blog는 Layout 설명, kernel 구성, 최적화 사례를 비교하는 데 사용합니다.

장별 source와 figure 원본은 [References](references/sources.md)에 기록합니다. 저장소에 포함된 third-party asset의 license는 [Third-party notices](THIRD_PARTY_NOTICES.md)에서 확인할 수 있습니다.
