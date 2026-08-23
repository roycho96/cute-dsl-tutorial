# 00. 이 책을 읽는 방법

CuTe DSL은 Python 문법으로 작성하지만, 일반적인 Python tensor library보다 CUDA에 가깝습니다. memory address, thread partition, instruction shape, synchronization을 직접 결정합니다. 반대로 CUTLASS C++의 방대한 template 계층을 먼저 배울 필요는 없습니다.

이 책은 두 성질을 모두 반영합니다. 처음 몇 장에서는 작은 숫자와 짧은 kernel만 사용합니다. Layout을 주소 함수로 읽을 수 있게 된 뒤에야 TMA와 Tensor Core를 도입합니다. 뒤쪽 장의 GEMM도 앞에서 만든 개념을 조합해서 설명합니다.

## 독자에게 필요한 배경

다음 코드를 읽을 수 있으면 시작할 수 있습니다.

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

다음 내용은 이미 알고 있다고 가정합니다.

- kernel launch와 grid·block·thread
- global, shared, register memory
- warp 단위 실행과 coalescing
- `__syncthreads()`가 필요한 이유
- 행렬 곱셈 `C = A @ B`

Tensor Core instruction, TMA, CUTLASS template, layout algebra는 처음부터 설명합니다.

## 환경 준비

CuTe DSL 4.6.1은 Linux에서 사용합니다. NVIDIA의 4.6 계열 source와 정확히 맞추려면 CUTLASS repository의 `setup.sh`를 사용하는 편이 안전합니다.

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# CUDA 13 계열
./python/CuTeDSL/setup.sh --cu13
```

stable wheel만 필요한 경우에는 다음처럼 설치할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "nvidia-cutlass-dsl[cu13]==4.6.1" torch
```

CUDA 12.9 환경에서는 `cu13` extra를 제외하고 설치합니다. wheel과 CUTLASS example source의 버전이 다르면 API 이름이나 동작 조건이 달라질 수 있으므로, 문제를 재현할 때는 두 버전을 함께 기록합니다.

설치 확인:

```bash
python - <<'PY'
import importlib.metadata as md
import torch

print("CuTe DSL:", md.version("nvidia-cutlass-dsl"))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("SM:", torch.cuda.get_device_capability(0))
PY
```

## 예제를 읽는 순서

각 장의 코드는 다음 순서로 확인합니다.

1. PyTorch reference가 계산하는 값을 확인합니다.
2. `@cute.jit` 함수가 어떤 type과 shape에 맞춰 compile되는지 봅니다.
3. `@cute.kernel` 안에서 thread가 맡는 좌표를 계산합니다.
4. 실제 load, compute, store가 어느 memory space에서 일어나는지 추적합니다.
5. correctness test를 먼저 통과시킨 뒤 성능을 측정합니다.

성능 수치는 개념 설명과 분리합니다. example은 API와 dataflow를 보여 주기 위한 코드이며, 별도 측정 없이 빠르다고 평가하지 않습니다.

## 표기

- `M`, `N`, `K`: GEMM의 논리 축
- `gA`, `sA`, `rA`: 각각 global, shared, register memory에 놓인 tensor A
- `tAgA`: partition `tA`를 global tensor `gA`에 적용한 thread-local view
- CTA: CUDA thread block
- RMEM, SMEM, GMEM, TMEM: register, shared, global, tensor memory

API와 변수 이름은 원문을 유지하고, 한국어 설명에서 그 역할을 명확히 적습니다. 번역했을 때 의미가 달라지는 `Layout`, `Tensor`, `TiledCopy`, `TiledMMA`, `atom`, `mode`는 영어 표기를 사용합니다.

다음 장에서는 이 코드가 Python process, JIT host function, GPU kernel의 세 영역을 어떻게 오가는지부터 확인합니다.

