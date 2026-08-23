import argparse

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


def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic()


def main(size: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")
    if size <= 0:
        raise ValueError("size는 양수여야 합니다.")

    torch.manual_seed(2026)
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    out = torch.empty_like(a)

    a_cute = as_cute_tensor(a)
    b_cute = as_cute_tensor(b)
    out_cute = as_cute_tensor(out)

    compiled = cute.compile(
        vector_add,
        a_cute,
        b_cute,
        out_cute,
        options="--generate-line-info",
    )
    compiled(a_cute, b_cute, out_cute)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, a + b, rtol=0, atol=0)
    print(f"PASS: {size} FP32 elements")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4099)
    args = parser.parse_args()
    main(args.size)
