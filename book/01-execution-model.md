# 01. CuTe DSL execution model

CuTe DSL 프로그램은 하나의 Python 파일 안에 host code와 device code를 함께 작성합니다. `@cute.jit`과 `@cute.kernel`은 단순한 Python decorator가 아닙니다. 두 decorator는 function이 compile되는 시점과 실행되는 위치, 호출 방식을 결정합니다.

![CuTe DSL program flow](../assets/01-execution-model.svg)

*Figure 1-1. Python driver에서 `@cute.jit` host function을 compile하고 `@cute.kernel`을 launch하는 과정.*

## `@cute.jit`과 `@cute.kernel`

| Code | Runs on | Responsibility |
|---|---|---|
| Python driver | CPU, Python runtime | tensor allocation, compile request, result validation |
| `@cute.jit` function | CPU, compiled host code | specialization, launch configuration |
| `@cute.kernel` function | GPU | load, compute, store |

`@cute.kernel`로 정의한 function은 Python에서 직접 실행하지 않습니다. `@cute.jit` host function이 kernel call을 만들고 `.launch()`에 grid와 block을 전달합니다. Python driver는 `cute.compile()`로 host function을 compile한 뒤 반환된 callable을 실행합니다.

가장 작은 vector addition으로 동작을 확인하겠습니다.

## Minimal example

전체 코드는 [`examples/01_execution_model/vector_add.py`](../examples/01_execution_model/vector_add.py)에 있습니다.

```python
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bid * bdim + tid

    if i < cute.size(out):
        out[i] = a[i] + b[i]


@cute.jit
def vector_add(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    threads = 256
    blocks = cute.ceil_div(cute.size(out), threads)
    vector_add_kernel(a, b, out).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )
```

CUDA C++ elementwise kernel과 thread index 계산은 같습니다. CUDA launch syntax 대신 `@cute.jit` function 안에서 kernel object의 `.launch()`를 호출한다는 점이 다릅니다.

## DLPack conversion does not copy data

Python driver는 PyTorch tensor를 DLPack을 통해 CuTe `Tensor`로 전달합니다.

```python
a_cute = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
b_cute = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
out_cute = from_dlpack(out, assumed_align=16).mark_layout_dynamic()
```

`from_dlpack()`은 device-to-device copy를 수행하지 않습니다. `torch.Tensor`와 CuTe `Tensor`가 같은 CUDA allocation을 참조합니다. Kernel이 `out_cute`에 기록한 결과를 PyTorch의 `out`에서 바로 읽을 수 있는 이유입니다.

`assumed_align=16`은 base address가 16-byte aligned라고 compiler에 알려 줍니다. 실제 pointer를 정렬하거나 allocation을 변경하지 않으므로 address가 이 조건을 만족할 때만 사용해야 합니다.

`mark_layout_dynamic()`은 runtime shape와 stride를 compiled function의 argument로 유지합니다. 반대로 static shape, Layout, `Constexpr` value는 specialization에 포함될 수 있습니다. Static argument와 dynamic argument의 차이는 Layout 장에서 구체적인 예제로 다룹니다.

## Compile and launch are separate operations

```python
compiled = cute.compile(vector_add, a_cute, b_cute, out_cute)
```

`cute.compile()`은 argument type과 static information을 바탕으로 host code와 device code를 생성합니다. 이 호출에서 발생하는 JIT compile time은 kernel latency가 아닙니다.

GPU kernel launch는 compiled function을 호출할 때 일어납니다.

```python
compiled(a_cute, b_cute, out_cute)
```

동일한 signature는 JIT cache를 재사용할 수 있습니다. `dtype`, static shape, Layout, `Constexpr`가 달라지면 별도의 specialization이 생성될 수 있습니다.

## Inside the compiler

CuTe DSL은 Python source를 바로 CUDA C++ source로 변환하지 않습니다. 기본 mode는 AST rewrite와 tracing을 함께 사용합니다.

![NVIDIA CuTe DSL compilation pipeline](../assets/nvidia-cute-dsl-compilation.png)

*Figure 1-2. Python AST rewrite, tracing, MLIR lowering, and binary generation in the CuTe DSL compiler.*

Compiler pipeline은 세 단계로 나눌 수 있습니다.

1. **AST rewrite**: Python control flow를 분석해 loop와 branch structure를 보존합니다.
2. **Tracing**: Proxy argument로 function을 실행하면서 CuTe operation을 MLIR에 기록합니다.
3. **Lowering and code generation**: CuTe dialect를 hardware-specific IR로 낮추고 GPU binary를 생성합니다.

`@cute.jit`의 기본값인 `preprocess=True`가 이 hybrid mode를 사용합니다. `preprocess=False`는 tracing만 수행합니다. Tracing-only mode에서는 실행되지 않은 branch가 기록되지 않고 loop가 trace 당시 iteration count만큼 unroll될 수 있으므로 straight-line code가 아니라면 주의해야 합니다.

이 구조를 알고 있으면 오류가 발생한 단계를 구분할 수 있습니다. Python exception, DSL compile error, generated kernel의 runtime error는 서로 다른 위치에서 확인해야 합니다.

## Launch configuration and predication

```python
threads = 256
blocks = cute.ceil_div(cute.size(out), threads)
```

`threads`는 CTA당 thread 수이고 `blocks`는 전체 tensor를 처리하는 CTA 수입니다. 예제의 기본값은 4099개 원소입니다.

```text
ceil(4099 / 256) = 17 CTAs
17 × 256         = 4352 threads
4352 - 4099      = 253 threads with i >= 4099
```

마지막 CTA의 일부 thread는 유효한 원소를 갖지 않으므로 predicate가 필요합니다.

```python
if i < cute.size(out):
    out[i] = a[i] + b[i]
```

Predicate를 제거하면 마지막 CTA에서 allocation 범위를 벗어난 memory access가 발생합니다.

## A Tensor is an Engine plus a Layout

예제에서는 `a[i]`처럼 1D index만 사용하지만 `a`는 pointer가 아니라 CuTe `Tensor`입니다. Tensor는 storage를 나타내는 `Engine`과 logical coordinate를 storage offset으로 변환하는 `Layout`을 결합합니다.

```text
logical coordinate i
        │
        ▼
     Layout(i)
        │
        ▼
storage offset → load or store
```

현재 Layout은 contiguous 1D mapping입니다. 이후에는 같은 Tensor interface로 matrix coordinate, hierarchical tile, swizzled shared memory를 표현합니다.

## Correctness check

```python
compiled(a_cute, b_cute, out_cute)
torch.cuda.synchronize()
torch.testing.assert_close(out, a + b, rtol=0, atol=0)
```

이 예제는 FP32 addition을 한 번 수행하므로 PyTorch reference와 exact match를 요구합니다. Reduction이나 GEMM처럼 연산 순서가 달라질 수 있는 kernel은 accumulation dtype과 numerical error를 반영해 tolerance를 정해야 합니다.

실행:

```bash
python examples/01_execution_model/vector_add.py
```

마지막 CTA의 predication과 여러 CTA를 함께 확인하려면 shape를 바꿔 실행합니다.

```bash
python examples/01_execution_model/vector_add.py --size 17
python examples/01_execution_model/vector_add.py --size 4096
python examples/01_execution_model/vector_add.py --size 4099
python examples/01_execution_model/vector_add.py --size 1048576
```

## Inspecting generated code

Generated IR을 terminal에 출력할 수 있습니다.

```bash
CUTE_DSL_PRINT_IR=1 \
python examples/01_execution_model/vector_add.py --size 17
```

IR, PTX, CUBIN을 파일로 저장하려면 다음 option을 사용합니다.

```bash
mkdir -p build/dump
CUTE_DSL_KEEP=ir,ptx,cubin \
CUTE_DSL_DUMP_DIR=build/dump \
python examples/01_execution_model/vector_add.py
```

`CUTE_DSL_KEEP`은 `ir-debug`, `sass`, `all`도 지원합니다. SASS를 생성하려면 DSL package의 `sass` extra 또는 CUBIN과 호환되는 `nvdisasm`이 필요합니다. Debug option은 JIT cache key와 generated code에 영향을 줄 수 있으므로 성능은 option을 끈 build로 다시 측정해야 합니다.

## Summary

- `@cute.jit`은 host function을 정의하고 `@cute.kernel`은 GPU kernel을 정의합니다.
- `cute.compile()`과 compiled function call은 각각 compile과 launch입니다.
- DLPack conversion은 CUDA allocation을 공유하며 data copy를 만들지 않습니다.
- CuTe `Tensor`는 `Engine`과 `Layout`으로 구성됩니다.
- 기본 compiler mode는 AST rewrite와 tracing을 함께 사용합니다.

다음 장에서는 `Shape`, `Stride`, `Layout`을 작은 정수 예제로 다룹니다. 목표는 GPU code를 작성하기 전에 `Layout(coord) = offset`을 직접 계산하는 것입니다.

## References

1. [NVIDIA, “DSL Programming Model: Introduction,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
2. [NVIDIA, “End-to-End Code Generation,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html) — source of Figure 1-2.
3. [NVIDIA, “Quick Start Guide,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
4. [NVIDIA, “Educational Notebooks,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/notebooks.html)
5. [NVIDIA, “Debugging,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html)
