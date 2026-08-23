# 00. 시작하기

CuTe DSL은 Python syntax를 사용하지만 PyTorch 같은 tensor framework는 아닙니다. 개발자가 memory layout, thread partition, copy instruction, MMA instruction, synchronization을 직접 선택합니다. CUDA C++보다 작성은 간결하지만 hardware에 대한 책임은 그대로 남습니다.

처음부터 GEMM 전체를 읽으면 `Layout`, `TiledCopy`, `TiledMMA`, pipeline state가 한꺼번에 등장합니다. 이 책은 다음 순서로 범위를 넓힙니다.

1. Python code가 언제 compile되고 어디에서 실행되는지 구분합니다.
2. 작은 정수 예제로 `Layout(coord) = offset`을 계산합니다.
3. Layout을 thread와 data에 적용해 copy와 MMA partition을 만듭니다.
4. GMEM, SMEM, RMEM 사이의 dataflow를 구성합니다.
5. TMA와 Tensor Core를 연결해 multistage GEMM을 완성합니다.
6. Hopper와 Blackwell의 architecture-specific instruction으로 확장합니다.

## 필요한 CUDA 지식

다음 kernel의 index 계산과 memory access를 설명할 수 있으면 시작하기에 충분합니다.

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

다음 내용은 알고 있다고 가정합니다.

- kernel launch와 grid, block, thread
- global memory, shared memory, register
- warp execution과 coalescing
- `__syncthreads()`가 필요한 경우
- matrix multiplication `C = A @ B`

Tensor Core, TMA, CUTLASS template, Layout algebra는 처음부터 설명합니다.

## Environment

이 저장소의 예제는 CuTe DSL 4.6.1에서 검증합니다. CUTLASS example과 동일한 version을 사용하려면 해당 CUTLASS checkout의 setup script로 설치하는 것이 가장 확실합니다.

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# CUDA 13
./python/CuTeDSL/setup.sh --cu13

# CUDA 12.9
./python/CuTeDSL/setup.sh --cu12
```

Stable wheel을 사용할 수도 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "nvidia-cutlass-dsl[cu13]==4.6.1" torch
```

CUDA 12.9에서는 `cu13` extra를 제외합니다. Wheel과 CUTLASS source의 version이 다르면 API나 compile behavior가 달라질 수 있으므로 문제를 재현할 때 두 version을 함께 기록합니다.

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

각 예제는 다음 순서로 읽는 것이 좋습니다.

1. PyTorch reference가 만드는 결과와 tensor shape를 확인합니다.
2. `@cute.jit` function의 static argument와 dynamic argument를 구분합니다.
3. `@cute.kernel`에서 각 thread가 담당하는 coordinate를 계산합니다.
4. GMEM, SMEM, RMEM에서 어떤 load와 store가 일어나는지 추적합니다.
5. correctness test를 통과시킨 뒤 profiler로 generated code를 확인합니다.

예제 코드는 개념과 dataflow를 보여 주기 위한 최소 구현입니다. 별도의 benchmark가 없는 예제에 성능 우위를 부여하지 않습니다.

## Naming convention

- `M`, `N`, `K`: GEMM의 logical axis
- `gA`, `sA`, `rA`: GMEM, SMEM, RMEM에 있는 tensor A
- `tAgA`: thread partition `tA`를 global tensor `gA`에 적용한 view
- CTA: CUDA thread block
- GMEM, SMEM, RMEM, TMEM: global, shared, register, tensor memory

`Layout`, `Tensor`, `TiledCopy`, `TiledMMA`, `atom`, `mode`, `pipeline`, `epilogue`처럼 CuTe code와 직접 대응하는 용어는 번역하지 않습니다.

다음 장에서는 `@cute.jit`과 `@cute.kernel`이 각각 언제 compile되고 실행되는지부터 살펴봅니다.
