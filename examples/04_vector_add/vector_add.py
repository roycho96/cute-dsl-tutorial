import argparse

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

THREADS = 256
VALUES_PER_THREAD = 4


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


@cute.jit
def scalar_vector_add(
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    blocks = cute.ceil_div(cute.size(out), THREADS)
    scalar_vector_add_kernel(a, b, out).launch(
        grid=(blocks, 1, 1),
        block=(THREADS, 1, 1),
    )


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
        grid=(blocks, 1, 1),
        block=(THREADS, 1, 1),
    )


def as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic()


def main(size: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if size <= 0:
        raise ValueError("size must be positive")

    torch.manual_seed(2026)
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
    print(f"PASS: {size} FP32 elements")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4099)
    args = parser.parse_args()
    main(args.size)
