# 06. Shared-memory transpose

행렬 transpose는 입력 `src[M, N]`을 출력 `dst[N, M]`으로 바꿉니다.

```text
dst[col, row] = src[row, col]
```

이 장에서는 32×32 tile을 shared memory에 올려 global memory의 load와 store를 모두 연속 접근으로 만듭니다.

![Shared-memory transpose](../assets/06-transpose.svg)

*Figure 6-1. 32×32 input tile을 32×33 shared-memory tile에 저장한 뒤 coordinate를 바꾸어 output tile에 쓴다.*

실행 가능한 전체 코드는 [`examples/06_transpose/transpose.py`](../examples/06_transpose/transpose.py)에 있습니다.

## 1. Grid와 block 정하기

입력의 row 수를 `M`, column 수를 `N`이라고 하겠습니다. Grid의 x축은 column 방향 tile을, y축은 row 방향 tile을 선택합니다.

```python
TILE_DIM = 32
BLOCK_ROWS = 8


@cute.jit
def transpose(
    src: cute.Tensor,
    dst: cute.Tensor,
):
    # Split an MxN matrix into 32x32 tiles along columns and rows.
    tile_cols = cute.ceil_div(cute.size(src, mode=[1]), TILE_DIM)
    tile_rows = cute.ceil_div(cute.size(src, mode=[0]), TILE_DIM)
    transpose_kernel(src, dst).launch(
        grid=(tile_cols, tile_rows, 1),  # x: ceil_div(N, 32), y: ceil_div(M, 32)
        block=(TILE_DIM, BLOCK_ROWS, 1),  # 32x8 threads; four values each
    )
```

Launch configuration을 식으로 쓰면 다음과 같습니다.

```text
grid.x  = ceil_div(N, 32)  # input column tiles
grid.y  = ceil_div(M, 32)  # input row tiles
block.x = 32               # one thread per tile column
block.y = 8                # eight active tile rows at a time
```

Block에는 `32 × 8 = 256`개 thread가 있습니다. 한 번에 8개 row를 처리하고 같은 thread가 row 방향으로 8씩 이동하므로, 각 thread는 최대 4개 값을 옮깁니다.

```text
threadIdx.y + row_offset = ty + {0, 8, 16, 24}
```

## 2. Shared-memory Layout

Shared-memory tile은 logical shape가 32×33이고 stride도 `(33, 1)`입니다.

```python
smem = cutlass.utils.SmemAllocator()
tile = smem.allocate_tensor(
    cutlass.Float32,
    cute.make_layout((TILE_DIM, TILE_DIM + 1), stride=(TILE_DIM + 1, 1)),
    byte_alignment=16,
)
```

```text
shape  = (32, 33)
stride = (33, 1)
offset(row, col) = row × 33 + col
```

실제 데이터는 각 row의 앞 32칸만 사용합니다. 마지막 한 칸은 row stride를 32에서 33으로 바꾸는 padding입니다.

Shared memory에는 32개 bank가 있고, 연속된 32-bit word는 연속된 bank에 대응합니다. Stride가 32이면 한 warp가 column 방향으로 읽을 때 모든 주소가 같은 bank에 놓입니다. Stride를 33으로 만들면 row가 바뀔 때 bank도 한 칸씩 이동하므로 이 접근의 32-way bank conflict를 피할 수 있습니다.

## 3. Tile을 읽고 쓰기

전체 kernel은 다음과 같습니다.

```python
@cute.kernel
def transpose_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
):
    tx, ty, _ = cute.arch.thread_idx()
    tile_x, tile_y, _ = cute.arch.block_idx()
    rows = cute.size(src, mode=[0])
    cols = cute.size(src, mode=[1])

    smem = cutlass.utils.SmemAllocator()
    tile = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((TILE_DIM, TILE_DIM + 1), stride=(TILE_DIM + 1, 1)),
        byte_alignment=16,
    )

    for row_offset in cutlass.range_constexpr(0, TILE_DIM, BLOCK_ROWS):
        row = tile_y * TILE_DIM + ty + row_offset
        col = tile_x * TILE_DIM + tx
        if row < rows and col < cols:
            tile[(ty + row_offset, tx)] = src[(row, col)]

    cute.arch.sync_threads()

    for row_offset in cutlass.range_constexpr(0, TILE_DIM, BLOCK_ROWS):
        out_row = tile_x * TILE_DIM + ty + row_offset
        out_col = tile_y * TILE_DIM + tx
        if out_row < cols and out_col < rows:
            dst[(out_row, out_col)] = tile[(tx, ty + row_offset)]
```

Load 단계에서 같은 `ty + row_offset`를 가진 32개 thread는 `tx=0..31`의 연속된 column을 읽습니다.

```text
global load: src[tile_y × 32 + ty + row_offset,
                 tile_x × 32 + tx]
shared write: tile[ty + row_offset, tx]
```

`sync_threads()` 뒤에는 shared-memory coordinate의 row와 column을 바꾸어 읽습니다.

```text
shared read:  tile[tx, ty + row_offset]
global store: dst[tile_x × 32 + ty + row_offset,
                  tile_y × 32 + tx]
```

Store에서도 `tx=0..31`이 output의 연속된 column을 선택합니다. Shared memory가 load와 store 사이에서 접근 순서를 바꾸는 중간 공간으로 작동합니다.

`M`이나 `N`이 32의 배수가 아니면 가장자리 tile의 일부 coordinate가 행렬 밖을 가리킵니다. Load와 store를 각각 검사해야 하는 이유는 transpose 뒤에 row와 column의 범위도 서로 바뀌기 때문입니다.

`range_constexpr(0, 32, 8)`은 compile time에 네 번의 반복으로 펼쳐집니다. 반복 횟수가 고정되어 있고 작을 때 사용하는 방식입니다.

## 4. Runtime Layout으로 받기

PyTorch의 contiguous 2D tensor를 복사 없이 CuTe Tensor로 넘깁니다.

```python
def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=1)
```

`mark_layout_dynamic(leading_dim=1)`은 두 번째 mode의 stride가 `1`임을 compiler에 명시하고, shape와 나머지 stride를 runtime 값으로 둡니다. 마지막 dimension이 contiguous인 row-major tensor에 해당합니다. `M=1` 또는 `N=1`이라 여러 mode의 stride가 `1`로 보이는 경우에도 어느 mode가 contiguous인지 명확합니다.

## 5. 실행하고 확인하기

```bash
source ~/workspace/.venv_wsl/bin/activate
python examples/06_transpose/transpose.py --rows 1000 --cols 769
```

```text
PASS: (1000, 769) -> (769, 1000)
```

예제는 정방행렬뿐 아니라 32의 배수가 아닌 직사각형 행렬을 사용하고, 결과를 `src.T`와 정확히 비교합니다.

## Summary

- Grid는 32×32 input tile을 x축과 y축으로 나눕니다.
- 32×8 thread가 네 번 반복해 32×32 tile을 옮깁니다.
- Shared memory에서 coordinate를 바꾸면 global load와 store를 모두 연속 접근으로 만들 수 있습니다.
- `(32, 33):(33, 1)` Layout의 한 칸 padding은 transpose read의 bank conflict를 피합니다.
- 가장자리 tile은 load와 store 양쪽에 predication이 필요합니다.

다음 장에서는 warp reduction과 block reduction을 사용해 row-wise softmax를 구현합니다.

## References

1. [NVIDIA, “An Efficient Matrix Transpose in CUDA C/C++”](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
2. [NVIDIA, “CUDA C++ Programming Guide,” Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
3. [NVIDIA, “CUDA C++ Best Practices Guide,” Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
4. [NVIDIA, `SmemAllocator`, CUTLASS source](https://github.com/NVIDIA/cutlass/blob/4ca61d0662bbc835c98dccfca022c8265edd9022/python/CuTeDSL/cutlass/utils/smem_allocator.py)
