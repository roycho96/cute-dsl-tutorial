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
        offset = 1 << (4 - step)
        value += cute.arch.shuffle_sync_down(value, offset=offset)
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


@cute.kernel
def warp_reduction_kernel(
    x: cute.Tensor,
    warp_sums: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid

    value = cutlass.Float32(0.0)
    if i < cute.size(x):
        value = x[i]
    value = warp_reduce_sum(value)

    lane = tid & (WARP_SIZE - 1)
    warp = tid // WARP_SIZE
    if lane == 0:
        warp_sums[bid * WARPS_PER_BLOCK + warp] = value


@cute.kernel
def block_reduction_kernel(
    x: cute.Tensor,
    block_sums: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid

    value = cutlass.Float32(0.0)
    if i < cute.size(x):
        value = x[i]

    smem = cutlass.utils.SmemAllocator()
    scratch = smem.allocate_tensor(cutlass.Float32, WARPS_PER_BLOCK + 1)
    value = block_reduce_sum(value, scratch, tid)

    if tid == 0:
        block_sums[bid] = value


@cute.jit
def launch_warp_reduction(
    x: cute.Tensor,
    warp_sums: cute.Tensor,
):
    # Split N values into blocks of 256; each block produces eight warp sums.
    blocks = cute.ceil_div(cute.size(x), THREADS)
    warp_reduction_kernel(x, warp_sums).launch(
        grid=(blocks, 1, 1),  # x: groups of 256 input values
        block=(THREADS, 1, 1),  # x: 256 threads, one value per thread
    )


@cute.jit
def launch_block_reduction(
    x: cute.Tensor,
    block_sums: cute.Tensor,
):
    # Split N values into blocks of 256; each block produces one sum.
    blocks = cute.ceil_div(cute.size(x), THREADS)
    block_reduction_kernel(x, block_sums).launch(
        grid=(blocks, 1, 1),  # x: one output sum per 256 input values
        block=(THREADS, 1, 1),  # x: 256 threads cooperate on one sum
    )


def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor).mark_layout_dynamic()


def main(size: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if size <= 0:
        raise ValueError("size must be positive")

    torch.manual_seed(2026)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    blocks = (size + THREADS - 1) // THREADS
    warp_sums = torch.empty(
        blocks * WARPS_PER_BLOCK,
        device="cuda",
        dtype=torch.float32,
    )
    block_sums = torch.empty(blocks, device="cuda", dtype=torch.float32)

    x_cute = as_cute_tensor(x)
    warp_sums_cute = as_cute_tensor(warp_sums)
    block_sums_cute = as_cute_tensor(block_sums)
    warp_fn = cute.compile(launch_warp_reduction, x_cute, warp_sums_cute)
    block_fn = cute.compile(launch_block_reduction, x_cute, block_sums_cute)

    warp_fn(x_cute, warp_sums_cute)
    block_fn(x_cute, block_sums_cute)
    torch.cuda.synchronize()

    padded = torch.zeros(blocks * THREADS, device="cuda", dtype=torch.float32)
    padded[:size] = x
    warp_reference = padded.reshape(-1, WARP_SIZE).sum(dim=1)
    block_reference = padded.reshape(-1, THREADS).sum(dim=1)
    torch.testing.assert_close(warp_sums, warp_reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(block_sums, block_reference, rtol=1e-5, atol=1e-5)
    print(f"PASS: {size} values, {blocks} blocks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4099)
    args = parser.parse_args()
    main(args.size)
