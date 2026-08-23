# 00. 시작하기

CuTe DSL은 Python 문법으로 GPU kernel을 작성하는 DSL입니다. PyTorch처럼 연산을 구성하고 실행을 맡기는 tensor framework와 달리, memory layout, thread 분할, copy instruction, MMA instruction, synchronization을 개발자가 직접 정합니다. CUDA C++보다 코드는 짧지만, 어떤 thread가 어떤 data를 처리할지는 여전히 명시해야 합니다.

처음부터 GEMM 예제를 읽으면 `Layout`, `TiledCopy`, `TiledMMA`, pipeline이 한꺼번에 등장합니다. 다음 순서로 범위를 넓혀 갑니다.

1. Python code가 compile되는 시점과 host code, device code를 구분합니다.
2. 작은 정수 예제로 `Layout(coord) = offset`을 계산합니다.
3. Layout으로 data 처리 범위를 thread별로 나누고 `TiledCopy`와 `TiledMMA`를 구성합니다.
4. GMEM, SMEM, RMEM 사이에서 data가 이동하는 과정을 추적합니다.
5. TMA와 Tensor Core를 multistage pipeline으로 연결해 GEMM을 완성합니다.
6. 앞에서 만든 GEMM을 Hopper와 Blackwell의 instruction으로 확장합니다.

## 필요한 CUDA 지식

아래 kernel에서 각 thread가 처리하는 index와 마지막 `if`가 필요한 이유를 설명할 수 있으면 충분합니다.

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

다음 내용은 알고 있다고 가정합니다.

- kernel launch와 grid, block, thread index
- global memory, shared memory, register의 차이
- warp execution과 coalescing
- `__syncthreads()`가 필요한 경우
- 행렬곱 `C = A @ B`

Tensor Core instruction, TMA, CUTLASS template, Layout algebra는 처음부터 설명합니다.

## 환경 구성

예제는 CuTe DSL 4.6.1에서 검증합니다. 먼저 Python 가상환경을 만들고 활성화합니다.

```bash
python -m venv .venv
source .venv/bin/activate
```

### CUTLASS source와 함께 설치

CUTLASS 구현과 공식 예제를 함께 보려면 `v4.6.1` tag를 checkout한 뒤 그 안의 `setup.sh`를 실행합니다. `setup.sh`는 해당 source와 맞는 CuTe DSL wheel을 설치합니다.

```bash
git clone --branch v4.6.1 --depth 1 https://github.com/NVIDIA/cutlass.git
cd cutlass
```

CUDA Toolkit 12.9:

```bash
./python/CuTeDSL/setup.sh --cu12
```

CUDA Toolkit 13.1:

```bash
./python/CuTeDSL/setup.sh --cu13
```

두 경우 모두 예제에서 사용할 PyTorch를 추가로 설치합니다.

```bash
python -m pip install torch
```

### PyPI wheel 설치

이 저장소의 예제만 실행한다면 CUDA Toolkit 버전에 맞는 wheel을 설치합니다.

CUDA Toolkit 12.9:

```bash
python -m pip install "nvidia-cutlass-dsl==4.6.1" torch
```

CUDA Toolkit 13.1:

```bash
python -m pip install "nvidia-cutlass-dsl[cu13]==4.6.1" torch
```

CUTLASS source와 설치된 wheel의 버전이 다르면 API나 compile 결과가 달라질 수 있습니다. 오류를 재현하거나 성능을 비교할 때는 CUTLASS commit, CuTe DSL 버전, CUDA Toolkit 버전을 함께 기록합니다.

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

각 예제는 다음 순서로 읽습니다.

1. PyTorch reference의 tensor shape와 결과를 확인합니다.
2. `@cute.jit` function의 static argument와 dynamic argument를 구분합니다.
3. `@cute.kernel`에서 각 thread가 담당하는 coordinate를 계산합니다.
4. GMEM, SMEM, RMEM에서 발생하는 load와 store를 추적합니다.
5. PyTorch reference 결과와 일치하는지 확인한 뒤 generated IR과 profiler 결과를 봅니다.

예제는 개념과 dataflow를 설명하기 위한 최소 구현입니다. 별도의 benchmark가 없다면 성능에 관한 결론을 내리지 않습니다.

## 표기 규칙

| 표기 | 의미 |
|---|---|
| `M`, `N`, `K` | GEMM의 논리 축 |
| `gA`, `sA`, `rA` | GMEM, SMEM, RMEM에 있는 tensor A |
| `tAgA` | thread partition `tA`를 global tensor `gA`에 적용한 view |
| CTA | CUDA thread block |
| GMEM, SMEM, RMEM, TMEM | global, shared, register, tensor memory |

`Layout`, `Tensor`, `TiledCopy`, `TiledMMA`, `atom`, `mode`, `pipeline`, `epilogue`처럼 CuTe API와 code에 직접 대응하는 용어는 영어 표기를 유지합니다.

다음 장에서는 `@cute.jit`과 `@cute.kernel`이 각각 언제 compile되고 실행되는지 살펴봅니다.

## References

1. [NVIDIA, “CuTe DSL Quick Start Guide,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
2. [NVIDIA, “CUTLASS 4.6.1,” GitHub Release](https://github.com/NVIDIA/cutlass/releases/tag/v4.6.1)
3. [NVIDIA, “DSL Programming Model: Introduction,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
