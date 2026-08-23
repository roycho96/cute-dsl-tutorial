# CuTe DSL Tutorial

CUDA의 실행 모델과 메모리 계층을 알고 있는 독자가 CuTe DSL로 실제 GPU kernel을 설계하기까지 필요한 내용을 한국어로 설명합니다.

목표는 API 목록을 만드는 것이 아닙니다. `Layout`이 좌표를 주소로 바꾸는 과정부터 시작해 thread-value partition, TMA pipeline, Tensor Core MMA, Blackwell TMEM과 `tcgen05`까지 하나의 흐름으로 연결합니다. 모든 주요 장에는 실행 가능한 코드와 기술 도식을 함께 둡니다.

![CuTe DSL의 실행 경계](assets/01-execution-model.svg)

## 읽기 전에

- CUDA kernel launch, grid·block·thread를 이해하고 있어야 합니다.
- global/shared/register memory와 coalescing, bank conflict를 알고 있으면 충분합니다.
- CUTLASS C++ template 경험은 필요하지 않습니다.
- 예제의 기준 버전은 Linux, Python 3.12, CuTe DSL 4.6.1입니다.

설치와 책의 사용 방법은 [00. 이 책을 읽는 방법](book/00-reading-guide.md)에서 설명합니다.

## 목차

### 1부. CuTe의 언어

- [x] [01. Python에서 GPU kernel까지](book/01-execution-model.md)
- [ ] 02. Shape와 IntTuple
- [ ] 03. Layout은 좌표 함수다
- [ ] 04. 계층적 Layout과 slicing
- [ ] 05. Layout algebra: coalesce와 composition
- [ ] 06. Layout algebra: complement, divide, tile

### 2부. Layout을 kernel로 옮기기

- [ ] 07. Tensor: Engine과 Layout
- [ ] 08. CTA tile과 local tile
- [ ] 09. Thread-value Layout
- [ ] 10. Copy atom과 vectorized copy
- [ ] 11. Shared-memory Layout과 swizzle
- [ ] 12. 경계 조건과 predicate

### 3부. GEMM의 구성 요소

- [ ] 13. MMA atom과 TiledMMA
- [ ] 14. SIMT GEMM에서 Tensor Core GEMM으로
- [ ] 15. GMEM→SMEM→RMEM dataflow
- [ ] 16. Epilogue와 output tile

### 4부. 비동기 pipeline

- [ ] 17. TMA tensor와 descriptor
- [ ] 18. `mbarrier`와 pipeline state
- [ ] 19. Multistage pipeline
- [ ] 20. Warp specialization
- [ ] 21. Persistent tile scheduler

### 5부. Hopper와 Blackwell

- [ ] 22. Hopper WGMMA
- [ ] 23. Blackwell TMEM과 `tcgen05`
- [ ] 24. Blackwell 1-SM GEMM
- [ ] 25. CTA pair와 2-SM MMA
- [ ] 26. TMA multicast와 cluster
- [ ] 27. NVFP4 block-scaled GEMM

### 6부. 실제 kernel

- [ ] 28. Fused epilogue
- [ ] 29. Grouped GEMM과 MoE
- [ ] 30. PyTorch·DLPack·AOT integration
- [ ] 31. 정확성 검증과 수치 오차
- [ ] 32. IR·PTX·SASS debugging
- [ ] 33. Nsight Compute로 병목 확인하기

## 코드 실행

공용 WSL 환경에서는 다음과 같이 실행합니다.

```bash
source ~/workspace/.venv_wsl/bin/activate
python examples/01_execution_model/vector_add.py
```

다른 환경의 설치 방법은 [00. 이 책을 읽는 방법](book/00-reading-guide.md#환경-준비)을 참고합니다.

## 자료 사용 원칙

기술 사실은 NVIDIA의 현재 문서와 CUTLASS source를 먼저 확인합니다. 논문과 Colfax를 비롯한 기술 블로그는 설명 순서와 구현 사례를 교차 검증하는 데 사용합니다. 재배포가 허용된 외부 그림은 원본 출처와 라이선스를 캡션에 밝히고, 그 밖의 도식과 예제는 이 책에 맞춰 새로 작성합니다. 장별 참고 자료와 확인 범위는 [출처 지도](references/sources.md)에 기록하며, 포함한 외부 자산은 [제3자 저작물 고지](THIRD_PARTY_NOTICES.md)에서 확인할 수 있습니다.
