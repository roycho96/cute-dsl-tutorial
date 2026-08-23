# 02. Shape and IntTuple

CUDA에서 행렬의 shape는 보통 `M × N`처럼 dimension을 나열해 표현합니다. CuTe의 `Shape`는 다릅니다. 각 mode가 다시 여러 mode를 포함할 수 있는 재귀 구조입니다.

```text
(2, 3, 4)       flat Shape
(2, (3, 4))     hierarchical Shape
```

두 Shape의 전체 원소 수는 24로 같지만 구조는 같지 않습니다. 이 차이는 뒤에서 CTA tile을 warp와 thread에 나누고, 각 thread가 처리할 값을 묶는 데 사용됩니다.

![Flat and hierarchical Shape](../assets/02-inttuple-shape.svg)

*Figure 2-1. `(2, 3, 4)`와 `(2, (3, 4))`는 size는 같지만 rank와 depth가 다르다.*

## IntTuple

CuTe에서 `IntTuple`은 integer 하나이거나, 다른 `IntTuple`을 원소로 갖는 tuple입니다.

```text
IntTuple := Integer | tuple[IntTuple, ...]
```

따라서 다음 값은 모두 `IntTuple`입니다.

```python
6
(6,)
(2, 3, 4)
(2, (3, 4))
((2, 4), (3, 5))
```

`Shape`과 coordinate는 `IntTuple` 형태를 사용하며, `Stride`도 같은 nested tuple 구조를 사용합니다. 같은 구조로 coordinate와 memory mapping을 기술할 수 있다는 점이 CuTe Layout algebra의 출발점입니다.

`Shape`에서 더 이상 tuple이 아닌 정수, 즉 leaf는 각 mode의 extent를 나타냅니다. 예를 들어 `(2, (3, 4))`는 바깥쪽에 두 mode가 있고, 두 번째 mode가 다시 extent 3과 4를 갖는 하위 구조입니다.

## Rank, depth, and size

CuTe는 Shape의 구조를 `rank`, `depth`, `size`로 구분합니다.

- `rank(x)`: 가장 바깥쪽에 있는 mode의 수
- `depth(x)`: tuple이 중첩된 최대 단계
- `size(x)`: 모든 leaf를 곱한 값

| Shape | `rank` | `depth` | `size` |
|---|---:|---:|---:|
| `6` | 1 | 0 | 6 |
| `(6,)` | 1 | 1 | 6 |
| `(2, 3, 4)` | 3 | 1 | 24 |
| `(2, (3, 4))` | 2 | 2 | 24 |

정수 하나는 rank가 1이고 depth가 0입니다. 원소가 하나인 tuple `(6,)`도 rank와 size는 같지만 depth가 1입니다. CuTe에서는 두 값의 계층이 다르므로 서로 같은 Shape으로 취급하면 안 됩니다.

`rank`는 모든 leaf의 개수를 세지 않습니다. `(2, (3, 4))`의 바깥쪽 mode는 `2`와 `(3, 4)` 두 개이므로 rank는 2입니다. 두 번째 mode만 선택하면 그 하위 Shape `(3, 4)`의 rank는 2입니다.

```python
shape = (2, (3, 4))

assert cute.rank(shape) == 2
assert cute.rank(shape, mode=[1]) == 2
assert cute.depth(shape) == 2
assert cute.size(shape) == 24
assert cute.size(shape, mode=[1]) == 12
```

## Selecting a mode

`cute.get()`은 mode 경로를 따라 Shape의 일부를 가져옵니다.

```python
shape = (2, (3, 4))

assert cute.get(shape, mode=[0]) == 2
assert cute.get(shape, mode=[1]) == (3, 4)
assert cute.get(shape, mode=[1, 0]) == 3
assert cute.get(shape, mode=[1, 1]) == 4
```

`mode=[1, 0]`은 바깥쪽 mode 1을 선택한 뒤, 그 안의 mode 0을 선택한다는 뜻입니다. 이 경로는 Shape의 구조를 가리킬 뿐 memory access 순서를 정하지 않습니다. Memory offset은 다음 장에서 `Stride`와 함께 결정합니다.

계층은 tile의 의미를 보존하는 데 유용합니다. 예를 들어 `(CTA, (warp, thread))`처럼 상위와 하위 분할을 하나의 Shape에 담을 수 있습니다. 실제 kernel에서는 숫자 대신 각 단계의 extent가 들어갑니다.

## Static and dynamic values

CuTe DSL에서 Python integer와 `Constexpr` argument는 compile time에 알려진 static value입니다. `cutlass.Int32` 같은 JIT argument는 compiled function을 호출할 때 전달되는 dynamic value입니다.

```python
@cute.jit
def inspect_shapes(m: cutlass.Int32, n: cutlass.Constexpr[int]):
    static_shape = (2, (3, n))
    mixed_shape = (m, (3, n))

    print(static_shape)
    print(mixed_shape)
    cute.printf("mixed at runtime: {}", mixed_shape)
```

`n=4`로 compile하면 `static_shape` 전체를 compile time에 계산할 수 있습니다. `mixed_shape`에는 dynamic value인 `m`이 있으므로 compile 과정의 `print()`에는 다음과 같이 표시됩니다.

```text
(2, (3, 4))
(Int32(?), (3, 4))
```

Compiled function을 실행할 때는 `cute.printf()`가 실제 값을 출력합니다.

```text
mixed at runtime: (5,(3,4))
```

`Constexpr` argument는 specialization에 포함되며 compiled function의 runtime signature에서는 빠집니다. 다음 compile 결과를 호출할 때 `n`을 다시 전달하지 않는 이유입니다.

```python
compiled = cute.compile(inspect_shapes, cutlass.Int32(5), 4)

compiled(cutlass.Int32(5))
compiled(cutlass.Int32(6))
```

두 호출은 같은 compiled callable을 사용하지만 runtime Shape는 각각 `(5, (3, 4))`와 `(6, (3, 4))`입니다. 반면 `n`을 다른 `Constexpr` value로 바꾸면 별도의 specialization이 필요합니다.

`Shape`을 static으로 두면 compiler가 loop bound와 Layout transform을 compile time에 계산할 수 있습니다. Dynamic Shape은 하나의 compiled function을 여러 runtime 크기에 재사용할 수 있게 합니다. 어느 쪽이 더 좋은지는 specialization cost와 generated code가 얻는 이점을 함께 측정해 결정해야 합니다.

## Running the example

전체 코드는 [`examples/02_shape_inttuple/shape_basics.py`](../examples/02_shape_inttuple/shape_basics.py)에 있습니다.

```bash
python examples/02_shape_inttuple/shape_basics.py --m 5 --n 4
```

주요 출력은 다음과 같습니다.

```text
shape             rank depth size
  6              1 0 6
  (6,)           1 1 6
  (2, 3, 4)      3 1 24
  (2, (3, 4))    2 2 24
mode [1]         (3, 4)
mode [1, 0]      3
mixed at compile time: (Int32(?), (3, 4))
mixed at runtime: (5,(3,4))
mixed size: 60
mixed at runtime: (6,(3,4))
mixed size: 72
```

마지막 두 쌍은 dynamic argument `m`만 바꾼 결과입니다. `n=4`는 compile time에 고정되어 있습니다.

## Shape does not determine memory order

Shape은 valid coordinate의 범위와 계층만 나타냅니다. `(2, 3)`이라는 Shape만 보고 row-major인지 column-major인지 알 수는 없습니다.

```text
Shape (2, 3)

row-major offsets:      0 1 2
                        3 4 5

column-major offsets:   0 2 4
                        1 3 5
```

두 배열은 Shape이 같고 size도 6이지만 coordinate를 memory offset으로 바꾸는 규칙이 다릅니다. CuTe에서는 이 규칙을 `Stride`로 표현하고, `Shape`와 `Stride`의 조합을 `Layout`이라고 부릅니다.

## Summary

- `IntTuple`은 integer와 nested tuple을 함께 표현하는 재귀 구조입니다.
- `rank`는 가장 바깥쪽의 mode 수, `depth`는 tuple이 중첩된 최대 단계, `size`는 모든 leaf의 곱입니다.
- `cute.get()`과 mode 경로로 hierarchical Shape의 일부를 선택할 수 있습니다.
- Python integer와 `Constexpr`는 static value이고 typed JIT argument는 dynamic value입니다.
- Shape은 coordinate domain을 나타내며 memory offset은 정하지 않습니다.

다음 장에서는 `Stride`를 도입해 `Layout(coord) = offset`을 직접 계산합니다. Row-major와 column-major Layout을 같은 Shape에서 만들고, coordinate가 memory offset으로 변환되는 과정을 확인합니다.

## References

1. [NVIDIA, “CuTe Layout,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
2. [NVIDIA, “CuTe Layout Algebra,” CuTe DSL Educational Notebook](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb)
3. [NVIDIA, “JIT Argument Generation,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.html)
4. [NVIDIA, “Dynamic Layouts,” CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html)
5. [Colfax Research, “A note on the algebra of CuTe Layouts”](https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf)
