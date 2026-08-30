# 04. Vector addition: scalar and vectorized kernels

이 장에서는 같은 vector addition을 scalar와 vectorized kernel로 작성합니다. 목표는 CuTe DSL에서 kernel을 정의하고, launch configuration을 만들고, Tensor Layout을 바꾸는 방법을 익히는 것입니다.

```text
out[i] = a[i] + b[i]
```

![Scalar and vectorized vector addition](../assets/04-vector-add.svg)

*Figure 4-1. Scalar kernel은 thread마다 FP32 값 하나를 처리하고, vectorized kernel은 연속된 네 값을 하나의 16-byte packet으로 처리한다.*

실행 가능한 전체 코드는 [`examples/04_vector_add/vector_add.py`](../examples/04_vector_add/vector_add.py)에 있습니다. 아래에서는 CuTe DSL과 직접 관련된 코드를 순서대로 살펴봅니다.

## 1. Scalar kernel

먼저 필요한 모듈과 상수를 선언합니다.

```python
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

THREADS = 256
VALUES_PER_THREAD = 4
```

`@cute.kernel`은 GPU에서 실행할 함수를 정의합니다.

```python
@cute.kernel
def scalar_vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid

    if i < cute.size(out):
        out[i] = a[i] + b[i]
```

`thread_idx()`와 `block_idx()`는 `(x, y, z)` index를 반환합니다. 이 kernel은 x축만 사용하므로 나머지 두 값은 `_`로 받습니다. `cute.size(out)`은 Tensor의 logical element 수이며, 마지막 block의 범위를 검사하는 데 사용합니다.

Kernel launch는 `@cute.jit` function에 작성합니다.

```python
@cute.jit
def scalar_vector_add(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    blocks = cute.ceil_div(cute.size(out), THREADS)
    scalar_vector_add_kernel(a, b, out).launch(
        grid=(blocks, 1, 1),  # (x, y, z): one-dimensional grid
        block=(THREADS, 1, 1),  # (x, y, z): one-dimensional block
    )
```

`cute.ceil_div(N, THREADS)`는 `N`개 element를 처리하는 데 필요한 block 수를 올림 계산합니다. `launch()`의 `grid`와 `block`은 CUDA와 같은 `(x, y, z)` 순서입니다. 이 예제는 1D launch이므로 y축과 z축의 크기를 1로 둡니다.

```text
grid.x  = ceil_div(N, 256)
block.x = 256
i       = blockIdx.x × 256 + threadIdx.x
```

## 2. 네 값을 하나의 packet으로 처리하기

Vectorized kernel은 thread마다 연속된 FP32 값 네 개를 처리합니다. 먼저 PyTorch tensor를 CuTe Tensor로 변환합니다.

```python
def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic()
```

- `from_dlpack()`은 CUDA allocation을 복사하지 않고 공유합니다.
- `assumed_align=16`은 입력 pointer가 16-byte aligned라는 조건을 compiler에 전달합니다.
- `mark_layout_dynamic()`은 Tensor 크기를 runtime 값으로 유지합니다.

`assumed_align`은 pointer를 정렬하지 않습니다. Storage offset이 있는 view를 전달한다면 실제 주소가 조건을 만족하는지 확인해야 합니다.

### Packet Layout

원래 Tensor `a`의 Layout이 `(N):(1)`이면 coordinate `i`는 그대로 `a[i]`의 offset입니다. Packet 크기를 4로 정하면 같은 index를 두 값으로 나눌 수 있습니다.

```text
packet_idx = i // 4
lane       = i % 4
i          = lane + 4 × packet_idx
```

여기서 `lane`은 warp lane ID가 아니라 packet 안의 위치 `0, 1, 2, 3`을 뜻합니다. 예를 들어 `i = 6`은 두 번째 packet의 세 번째 값이므로 `(lane=2, packet_idx=1)`이 됩니다.

`zipped_divide()`가 이 coordinate 변환을 Layout에 적용합니다.

```python
packets_a = cute.zipped_divide(a, (VALUES_PER_THREAD,))
```

```text
(N):(1)
    ↓ zipped_divide(..., (4,))
((4),(?)):((1),(4))
```

변환된 Layout은 다음과 같이 읽습니다.

| Layout 항목 | 의미 |
|---|---|
| 첫 번째 shape `(4)` | packet 안에 네 값이 있습니다. |
| 첫 번째 stride `(1)` | `lane`이 1 증가하면 원래 offset도 1 증가합니다. |
| 두 번째 shape `(?)` | packet 수는 runtime의 `N`에서 결정됩니다. |
| 두 번째 stride `(4)` | `packet_idx`가 1 증가하면 원래 offset은 4 증가합니다. |

따라서 coordinate는 다음처럼 대응합니다.

| 원래 index `i` | Packet coordinate | CuTe access |
|---:|---:|---|
| 0 | `(0, 0)` | `packets_a[(0, 0)]` |
| 3 | `(3, 0)` | `packets_a[(3, 0)]` |
| 4 | `(0, 1)` | `packets_a[(0, 1)]` |
| 6 | `(2, 1)` | `packets_a[(2, 1)]` |

Integer로 두 mode를 모두 선택하면 값 하나를 얻습니다. 첫 번째 coordinate에 `None`을 쓰면 packet 내부 mode를 남긴 Tensor view를 얻습니다.

```python
packets_a[(2, 1)]     # a[6]
packets_a[(None, 1)]  # a[4:8]에 대응하는 네 값의 view
```

`N`이 4의 배수가 아니면 마지막 packet의 일부 coordinate는 allocation 밖을 가리킵니다. 예를 들어 `N = 10`이면 마지막 packet은 index `8, 9, 10, 11`에 대응하지만 `8, 9`만 유효합니다. 아래 kernel이 완전한 packet과 tail을 나누는 이유입니다.

### Vectorized kernel

```python
@cute.kernel
def vectorized_vector_add_kernel(
    packets_a: cute.Tensor,
    packets_b: cute.Tensor,
    packets_out: cute.Tensor,
    size: cutlass.Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    packet_idx = bid * THREADS + tid
    full_packets = size // VALUES_PER_THREAD

    if packet_idx < full_packets:
        packet_a = packets_a[(None, packet_idx)]
        packet_b = packets_b[(None, packet_idx)]
        packet_out = packets_out[(None, packet_idx)]
        packet_out.store(packet_a.load() + packet_b.load())
    elif packet_idx == full_packets:
        for lane in cutlass.range_constexpr(VALUES_PER_THREAD):
            i = packet_idx * VALUES_PER_THREAD + lane
            if i < size:
                packets_out[(lane, packet_idx)] = (
                    packets_a[(lane, packet_idx)] + packets_b[(lane, packet_idx)]
                )
```

`packets_a[(None, packet_idx)]`에서 `None`은 첫 번째 mode를 남깁니다. 따라서 결과는 연속된 네 값을 가리키는 Tensor view입니다. `load()`는 이 값을 `TensorSSA`로 읽고, `store()`는 계산 결과를 다시 씁니다.

마지막 1~3개 element는 완전한 16-byte packet이 아닙니다. 이 부분만 `range_constexpr()`로 펼친 scalar path에서 처리합니다.

### Vectorized launch

```python
@cute.jit
def vectorized_vector_add(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    packets_a = cute.zipped_divide(a, (VALUES_PER_THREAD,))
    packets_b = cute.zipped_divide(b, (VALUES_PER_THREAD,))
    packets_out = cute.zipped_divide(out, (VALUES_PER_THREAD,))
    size = cute.size(out)
    packets = cute.ceil_div(size, VALUES_PER_THREAD)
    blocks = cute.ceil_div(packets, THREADS)
    vectorized_vector_add_kernel(packets_a, packets_b, packets_out, size).launch(
        grid=(blocks, 1, 1),  # (x, y, z): one-dimensional grid
        block=(THREADS, 1, 1),  # (x, y, z): one-dimensional block
    )
```

Scalar version에서는 thread가 element 하나를 담당했습니다. 여기서는 thread가 packet 하나를 담당하므로 launch할 thread 수를 `ceil_div(N, 4)`에서 계산합니다.

## 3. Compile하고 실행하기

PyTorch tensor를 만든 뒤 `cute.compile()`에 JIT function과 예제 argument를 전달합니다.

```python
a = torch.randn(size, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
scalar_out = torch.empty_like(a)
vectorized_out = torch.empty_like(a)

a_cute = as_cute_tensor(a)
b_cute = as_cute_tensor(b)
scalar_out_cute = as_cute_tensor(scalar_out)
vectorized_out_cute = as_cute_tensor(vectorized_out)

scalar = cute.compile(scalar_vector_add, a_cute, b_cute, scalar_out_cute)
vectorized = cute.compile(
    vectorized_vector_add,
    a_cute,
    b_cute,
    vectorized_out_cute,
)

scalar(a_cute, b_cute, scalar_out_cute)
vectorized(a_cute, b_cute, vectorized_out_cute)
torch.cuda.synchronize()

reference = a + b
torch.testing.assert_close(scalar_out, reference, rtol=0, atol=0)
torch.testing.assert_close(vectorized_out, reference, rtol=0, atol=0)
```

`cute.compile()`은 compile한 callable을 반환합니다. 실제 실행에서는 같은 Layout과 dtype 조건을 만족하는 CuTe Tensor를 넘깁니다. Kernel launch는 비동기이므로 결과를 CPU에서 검사하기 전에 `torch.cuda.synchronize()`로 완료를 기다립니다.

```bash
source ~/workspace/.venv_wsl/bin/activate
python examples/04_vector_add/vector_add.py --size 4099
```

```text
PASS: 4099 FP32 elements
```

## 4. 생성된 memory instruction 확인하기

```bash
mkdir -p build/04_vector_add
CUTE_DSL_KEEP=ptx,cubin,sass \
CUTE_DSL_DUMP_DIR=build/04_vector_add \
python examples/04_vector_add/vector_add.py --size 4096
```

CuTe DSL 4.6.1과 `sm_120`에서 확인한 결과는 다음과 같습니다.

| Path | PTX | SASS |
|---|---|---|
| Scalar | `ld.global.b32` / `st.global.b32` | `LDG.E` / `STG.E` |
| Full packet | `ld.global.v2.b64` / `st.global.v2.b64` | `LDG.E.128` / `STG.E.128` |
| Tail | scalar load/store | `LDG.E` / `STG.E` |

`Tensor.load()`와 `Tensor.store()`가 항상 vector instruction을 만든다고 가정하면 안 됩니다. Alignment, Layout, target architecture가 바뀌면 생성 결과도 달라질 수 있으므로 PTX와 SASS를 확인합니다.

## Summary

- `@cute.kernel`은 GPU kernel을 정의합니다. 이 예제의 `@cute.jit` function은 Layout을 바꾸고 kernel을 launch합니다.
- `launch(grid=..., block=...)`은 `(x, y, z)` 순서로 실행 구성을 받습니다.
- `zipped_divide()`와 `None` slicing으로 thread 하나가 처리할 packet view를 만들 수 있습니다.
- `from_dlpack()`과 `cute.compile()`로 PyTorch tensor를 CuTe DSL kernel에 연결합니다.

다음 장에서는 warp와 block reduction을 구현합니다.

## References

1. [NVIDIA, “Educational Notebooks,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/notebooks.html)
2. [NVIDIA, “CuTe DSL Core API,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html)
3. [NVIDIA, “CUDA C++ Programming Guide,” Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
4. [NVIDIA, “Debugging CuTe DSL,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html)
5. [Simon Veitner, “An applied introduction to CuTeDSL”](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/)
