# CuTe DSL Tutorial

CUDA kernel을 작성해 본 독자를 위한 한국어 CuTe DSL 교재입니다. 목표는 CuTe의 출력에서 `Shape:Stride`를 읽고, thread와 data의 mapping을 추적하고, elementwise kernel과 reduction에서 Blackwell GEMM, FlashAttention, grouped MoE kernel까지 직접 구현하는 것입니다.

Part 1에서 `Shape`, `Layout`, `Tensor`를 세 장 안에 정리합니다. Part 2부터는 매 장마다 실행 가능한 kernel을 확장하면서 새로운 개념을 도입합니다. Layout algebra는 실제 코드에 필요한 순서로 설명하고, 세부 연산은 부록에 모읍니다.

![From Python to a GPU kernel](assets/01-execution-model.svg)

## 필요한 배경

- CUDA grid, block, thread, warp
- global memory, shared memory, register
- coalescing과 bank conflict
- `__syncthreads()`와 기본적인 synchronization
- 행렬곱 `C = A @ B`

CUTLASS C++ template이나 Tensor Core instruction을 미리 알 필요는 없습니다. 예제는 Linux, Python 3.12, CuTe DSL 4.6.1에서 검증합니다.

설치와 코드를 읽는 순서는 [00. 시작하기](book/00-reading-guide.md)에 정리했습니다.

## Learning path

### Part 1. CuTe 기본 개념

- [x] [01. First kernel and execution model](book/01-execution-model.md)
- [x] [02. Shape, Stride, and Layout](book/02-shape-stride-layout.md)
- [x] [03. Tensor, slicing, and tiling](book/03-tensor-slicing-tiling.md)

Part 1을 마치면 `(M, N):(N, 1)` 같은 Layout을 읽고, Tensor slicing과 tiling이 base offset과 Layout을 어떻게 바꾸는지 계산할 수 있어야 합니다.

### Part 2. Fundamental GPU kernels

- [x] [04. Vector addition: scalar and vectorized kernels](book/04-vector-add.md)
- [x] [05. Warp and block reduction](book/05-reduction.md)
- [x] [06. Shared-memory transpose and swizzle](book/06-shared-memory-transpose.md)
- [x] [07. Row-wise softmax](book/07-row-softmax.md)

Vector addition, reduction, transpose, softmax를 직접 구현합니다. Vectorized memory access, warp shuffle, shared-memory Layout과 swizzle, synchronization, predication, runtime loop를 필요한 시점에 도입합니다.

### Part 3. Building a Tensor Core GEMM

- [x] [08. MMA atom and TiledMMA](book/08-mma-atom-tiledmma.md)
- [x] [09. First tiled Tensor Core GEMM](book/09-first-tensor-core-gemm.md)
- [x] [10. Multistage GEMM and epilogue](book/10-multistage-gemm-epilogue.md)

Part 2에서 사용한 thread partitioning, vectorized memory access, shared-memory staging을 BF16 Tensor Core GEMM에 적용합니다. `m16n8k16` MMA atom에서 시작해 단일-stage GEMM, 3-stage `cp.async` mainloop, shared-memory epilogue까지 코드로 연결합니다.

Part 3을 마치면 `TiledMMA`의 warp·lane mapping을 읽고, GMEM → SMEM → RMEM → MMA → epilogue dataflow를 kernel 코드에서 추적할 수 있어야 합니다.

### Part 4. Asynchronous pipelines and modern architectures

- [ ] 11. TMA and `mbarrier`
- [ ] 12. Warp-specialized and persistent pipelines
- [ ] 13. Hopper WGMMA
- [ ] 14. Blackwell TMEM and `tcgen05`
- [ ] 15. Thread block clusters and 2-SM MMA

Part 3의 GEMM에 TMA pipeline과 Hopper·Blackwell instruction을 차례로 적용합니다. 두 architecture에서 memory와 execution model이 어떻게 달라지는지 code로 비교합니다.

### Part 5. Complete GEMM kernels

- [ ] 16. Blackwell GEMM end to end
- [ ] 17. NVFP4 block-scaled GEMM

앞에서 만든 구성 요소를 완전한 dense GEMM으로 조립한 뒤 NVFP4 block scaling을 적용합니다.

### Part 6. FlashAttention in CuTe DSL

- [ ] 18. FlashAttention-1: IO-aware tiling and online softmax
- [ ] 19. FlashAttention-2: work partitioning and sequence-dimension parallelism
- [ ] 20. FlashAttention-3: Hopper asynchronous pipeline
- [ ] 21. FlashAttention-4: Blackwell pipeline and conditional rescaling

네 장은 같은 Q/K/V interface와 correctness test를 사용합니다. FA1에서 attention matrix를 저장하지 않는 tiled forward를 구현하고, FA2의 thread-block·warp partitioning, FA3의 TMA·WGMMA pipeline, FA4의 TMEM·asynchronous MMA, software-emulated exponential과 conditional rescaling을 차례로 반영합니다. 코드는 현재 FlashAttention-4와 같은 CuTe DSL 구성 방식을 사용하되, 각 장에서 한 세대의 핵심 변화만 추가합니다.

### Part 7. Grouped kernels and MoE

- [ ] 22. Grouped GEMM and MoE case study

Dense kernel에서 여러 expert의 서로 다른 token 수를 처리하는 grouped scheduling으로 확장하고, routing 결과가 GEMM shape와 load balancing에 미치는 영향을 확인합니다.

### Appendices

- [ ] A. Layout algebra reference
- [ ] B. Correctness and numerical error
- [ ] C. Reading IR, PTX, and SASS
- [ ] D. Profiling with Nsight Compute
- [ ] E. PyTorch, DLPack, and AOT integration

## 예제 실행

```bash
source ~/workspace/.venv_wsl/bin/activate
python examples/01_execution_model/vector_add.py
```

## References and figures

기술적인 설명은 현재 NVIDIA documentation과 CUTLASS source를 기준으로 작성합니다. Colfax Research를 비롯한 논문과 technical blog는 Layout 설명, kernel 구성, 최적화 사례를 비교하는 데 사용합니다.

장별 source와 figure 원본은 [References](references/sources.md)에 기록합니다. 저장소에 포함된 third-party asset의 license는 [Third-party notices](THIRD_PARTY_NOTICES.md)에서 확인할 수 있습니다.
