import argparse

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

WARP_SIZE = 32
THREADS = 256
WARPS_PER_BLOCK = THREADS // WARP_SIZE


@cute.jit
def warp_reduce_sum(value: cute.Numeric) -> cute.Numeric:
    for step in range(5):
        offset = 1 << step
        value += cute.arch.shuffle_sync_bfly(value, offset=offset)
    return value


@cute.jit
def warp_reduce_max(value: cute.Numeric) -> cute.Numeric:
    for step in range(5):
        offset = 1 << step
        other = cute.arch.shuffle_sync_bfly(value, offset=offset)
        value = cute.arch.fmax(value, other)
    return value


@cute.jit
def block_reduce_sum(
    value: cute.Numeric,
    scratch: cute.Tensor,
    tid: cutlass.Int32,
) -> cute.Numeric:
    lane = tid & (WARP_SIZE - 1)
    warp = tid // WARP_SIZE
    value = warp_reduce_sum(value)
    if lane == 0:
        scratch[warp] = value
    cute.arch.sync_threads()

    if warp == 0:
        value = cutlass.Float32(0.0)
        if lane < WARPS_PER_BLOCK:
            value = scratch[lane]
        value = warp_reduce_sum(value)
        if lane == 0:
            scratch[WARPS_PER_BLOCK] = value
    cute.arch.sync_threads()
    return scratch[WARPS_PER_BLOCK]


@cute.jit
def block_reduce_max(
    value: cute.Numeric,
    scratch: cute.Tensor,
    tid: cutlass.Int32,
) -> cute.Numeric:
    lane = tid & (WARP_SIZE - 1)
    warp = tid // WARP_SIZE
    value = warp_reduce_max(value)
    if lane == 0:
        scratch[warp] = value
    cute.arch.sync_threads()

    if warp == 0:
        value = cutlass.Float32(-float("inf"))
        if lane < WARPS_PER_BLOCK:
            value = scratch[lane]
        value = warp_reduce_max(value)
        if lane == 0:
            scratch[WARPS_PER_BLOCK] = value
    cute.arch.sync_threads()
    return scratch[WARPS_PER_BLOCK]


@cute.kernel
def row_softmax_kernel(
    x: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    cols = cute.size(x, mode=[1])

    smem = cutlass.utils.SmemAllocator()
    scratch = smem.allocate_tensor(cutlass.Float32, WARPS_PER_BLOCK + 1)

    local_max = cutlass.Float32(-float("inf"))
    for col in cutlass.range(tid, cols, THREADS, unroll=1):
        local_max = cute.arch.fmax(local_max, x[(row, col)])
    row_max = block_reduce_max(local_max, scratch, tid)

    local_sum = cutlass.Float32(0.0)
    for col in cutlass.range(tid, cols, THREADS, unroll=1):
        local_sum += cute.math.exp(x[(row, col)] - row_max)
    row_sum = block_reduce_sum(local_sum, scratch, tid)

    for col in cutlass.range(tid, cols, THREADS, unroll=1):
        out[(row, col)] = cute.math.exp(x[(row, col)] - row_max) / row_sum


@cute.jit
def row_softmax(
    x: cute.Tensor,
    out: cute.Tensor,
):
    # Launch one 256-thread block for each of the M rows.
    rows = cute.size(x, mode=[0])
    row_softmax_kernel(x, out).launch(
        grid=(rows, 1, 1),  # x: M row blocks; y and z are unused
        block=(THREADS, 1, 1),  # x: columns tid, tid+256, ...
    )


def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def main(rows: int, cols: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    torch.manual_seed(2026)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32) * 4.0
    out = torch.empty_like(x)
    x_cute = as_cute_tensor(x)
    out_cute = as_cute_tensor(out)
    softmax_fn = cute.compile(row_softmax, x_cute, out_cute)

    softmax_fn(x_cute, out_cute)
    torch.cuda.synchronize()
    reference = torch.softmax(x, dim=1)
    torch.testing.assert_close(out, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        out.sum(dim=1),
        torch.ones(rows, device="cuda"),
        rtol=2e-5,
        atol=2e-6,
    )
    print(f"PASS: {rows} rows x {cols} columns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--cols", type=int, default=769)
    args = parser.parse_args()
    main(args.rows, args.cols)
