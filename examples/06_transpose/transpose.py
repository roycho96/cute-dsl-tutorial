import argparse

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

TILE_DIM = 32
BLOCK_ROWS = 8


@cute.jit
def copy_transposed_tile(
    src: cute.Tensor,
    dst: cute.Tensor,
    tile: cute.Tensor,
):
    tx, ty, _ = cute.arch.thread_idx()
    tile_x, tile_y, _ = cute.arch.block_idx()
    rows = cute.size(src, mode=[0])
    cols = cute.size(src, mode=[1])

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


@cute.kernel
def padding_transpose_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
):
    smem = cutlass.utils.SmemAllocator()
    tile = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((TILE_DIM, TILE_DIM + 1), stride=(TILE_DIM + 1, 1)),
        byte_alignment=16,
    )
    copy_transposed_tile(src, dst, tile)


@cute.kernel
def swizzled_transpose_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
):
    smem = cutlass.utils.SmemAllocator()
    base_layout = cute.make_layout(
        (TILE_DIM, TILE_DIM),
        stride=(TILE_DIM, 1),
    )
    # XOR row bits into bank bits without allocating padding.
    swizzled_layout = cute.make_composed_layout(
        cute.make_swizzle(5, 0, 5),
        0,
        base_layout,
    )
    tile = smem.allocate_tensor(
        cutlass.Float32,
        swizzled_layout,
        byte_alignment=16,
    )
    copy_transposed_tile(src, dst, tile)


@cute.jit
def padding_transpose(
    src: cute.Tensor,
    dst: cute.Tensor,
):
    # Split an MxN matrix into 32x32 tiles along columns and rows.
    tile_cols = cute.ceil_div(cute.size(src, mode=[1]), TILE_DIM)
    tile_rows = cute.ceil_div(cute.size(src, mode=[0]), TILE_DIM)
    padding_transpose_kernel(src, dst).launch(
        grid=(tile_cols, tile_rows, 1),  # x: ceil_div(N, 32), y: ceil_div(M, 32)
        block=(TILE_DIM, BLOCK_ROWS, 1),  # 32x8 threads; four values each
    )


@cute.jit
def swizzled_transpose(
    src: cute.Tensor,
    dst: cute.Tensor,
):
    # Split an MxN matrix into 32x32 tiles along columns and rows.
    tile_cols = cute.ceil_div(cute.size(src, mode=[1]), TILE_DIM)
    tile_rows = cute.ceil_div(cute.size(src, mode=[0]), TILE_DIM)
    swizzled_transpose_kernel(src, dst).launch(
        grid=(tile_cols, tile_rows, 1),  # x: ceil_div(N, 32), y: ceil_div(M, 32)
        block=(TILE_DIM, BLOCK_ROWS, 1),  # 32x8 threads; four values each
    )


def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def main(rows: int, cols: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    torch.manual_seed(2026)
    src = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    padding_out = torch.empty(cols, rows, device="cuda", dtype=torch.float32)
    swizzled_out = torch.empty_like(padding_out)
    src_cute = as_cute_tensor(src)
    padding_out_cute = as_cute_tensor(padding_out)
    swizzled_out_cute = as_cute_tensor(swizzled_out)
    padding_fn = cute.compile(padding_transpose, src_cute, padding_out_cute)
    swizzled_fn = cute.compile(swizzled_transpose, src_cute, swizzled_out_cute)

    padding_fn(src_cute, padding_out_cute)
    swizzled_fn(src_cute, swizzled_out_cute)
    torch.cuda.synchronize()
    torch.testing.assert_close(padding_out, src.T, rtol=0, atol=0)
    torch.testing.assert_close(swizzled_out, src.T, rtol=0, atol=0)
    print(f"PASS: padding and swizzle, ({rows}, {cols}) -> ({cols}, {rows})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--cols", type=int, default=769)
    args = parser.parse_args()
    main(args.rows, args.cols)
