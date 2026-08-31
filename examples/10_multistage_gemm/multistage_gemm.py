import argparse
import math

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

CTA_M = 64
CTA_N = 64
CTA_K = 32
MMA_SHAPE_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 2, 1)
THREADS = 128
STAGES = 3
COPY_BITS = 128


@cute.jit
def make_ab_smem_layout(rows: int):
    swizzle_bits = int(math.log2(CTA_K * cutlass.BFloat16.width // COPY_BITS))
    base = cute.make_layout((8, CTA_K), stride=(CTA_K, 1))
    atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        base,
    )
    return cute.tile_to_shape(atom, (rows, CTA_K, STAGES), (0, 1, 2))


@cute.jit
def make_c_smem_layout():
    # Keep the teaching epilogue flat so RMEM and vector-copy views stay visible.
    return cute.make_layout((CTA_M, CTA_N), stride=(CTA_N, 1))


@cute.jit
def make_tiled_copy(atom: cute.CopyAtom, cols: int):
    values_per_copy = COPY_BITS // cutlass.BFloat16.width
    threads_along_cols = cols // values_per_copy
    thread_layout = cute.make_layout(
        (THREADS // threads_along_cols, threads_along_cols),
        stride=(threads_along_cols, 1),
    )
    value_layout = cute.make_layout((1, values_per_copy))
    return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)


@cute.kernel
def multistage_gemm_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    sC_layout: cute.Layout,
    gmem_tiled_copy: cute.TiledCopy,
    c_tiled_copy: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    gA = cute.local_tile(a, (CTA_M, CTA_K), (tile_m, None))
    gB = cute.local_tile(b, (CTA_N, CTA_K), (tile_n, None))
    gC = cute.local_tile(c, (CTA_M, CTA_N), (tile_m, tile_n))
    gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
    gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
    gC = cute.make_tensor(gC.iterator.align(16), gC.layout)

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout, byte_alignment=16)
    sC = smem.allocate_tensor(cutlass.BFloat16, sC_layout, byte_alignment=16)

    thr_copy = gmem_tiled_copy.get_slice(tid)
    tAgA = thr_copy.partition_S(gA)
    tAsA = thr_copy.partition_D(sA)
    tBgB = thr_copy.partition_S(gB)
    tBsB = thr_copy.partition_D(sB)

    thr_mma = tiled_mma.get_slice(tid)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCsC = thr_mma.partition_C(sC)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    ldmatrix_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        cutlass.BFloat16,
    )
    s2r_copy_A = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma)
    s2r_copy_B = cute.make_tiled_copy_B(ldmatrix_atom, tiled_mma)
    thr_s2r_A = s2r_copy_A.get_slice(tid)
    thr_s2r_B = s2r_copy_B.get_slice(tid)
    tCsA_copy = thr_s2r_A.partition_S(sA)
    tCsB_copy = thr_s2r_B.partition_S(sB)
    tCrA_copy = thr_s2r_A.retile(tCrA)
    tCrB_copy = thr_s2r_B.retile(tCrB)

    k_tiles = cute.size(gA, mode=[2])
    k_blocks = cute.size(tCrA, mode=[2])

    # Fill stage 0 and stage 1 before entering the steady-state loop.
    for stage in cutlass.range_constexpr(STAGES - 1):
        if stage < k_tiles:
            cute.copy(
                gmem_tiled_copy,
                tAgA[None, None, None, stage],
                tAsA[None, None, None, stage],
            )
            cute.copy(
                gmem_tiled_copy,
                tBgB[None, None, None, stage],
                tBsB[None, None, None, stage],
            )
            cute.arch.cp_async_commit_group()

    read_stage = cutlass.Int32(0)
    write_stage = cutlass.Int32(STAGES - 1)
    next_k_tile = cutlass.Int32(STAGES - 1)

    for k_tile in cutlass.range(k_tiles, unroll=1):
        has_next_tile = next_k_tile < k_tiles
        if has_next_tile:
            # Keep one newer group in flight while the oldest stage is consumed.
            cute.arch.cp_async_wait_group(STAGES - 2)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if has_next_tile:
            cute.copy(
                gmem_tiled_copy,
                tAgA[None, None, None, next_k_tile],
                tAsA[None, None, None, write_stage],
            )
            cute.copy(
                gmem_tiled_copy,
                tBgB[None, None, None, next_k_tile],
                tBsB[None, None, None, write_stage],
            )
            cute.arch.cp_async_commit_group()
            next_k_tile += 1

        # Load k-block 0 before overlapping later ldmatrix loads with MMA.
        cute.copy(
            s2r_copy_A,
            tCsA_copy[None, None, 0, read_stage],
            tCrA_copy[None, None, 0],
        )
        cute.copy(
            s2r_copy_B,
            tCsB_copy[None, None, 0, read_stage],
            tCrB_copy[None, None, 0],
        )

        for k_block in cutlass.range_constexpr(k_blocks):
            if k_block + 1 < k_blocks:
                cute.copy(
                    s2r_copy_A,
                    tCsA_copy[None, None, k_block + 1, read_stage],
                    tCrA_copy[None, None, k_block + 1],
                )
                cute.copy(
                    s2r_copy_B,
                    tCsB_copy[None, None, k_block + 1, read_stage],
                    tCrB_copy[None, None, k_block + 1],
                )
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_block],
                tCrB[None, None, k_block],
                tCrC,
            )

        # No thread may overwrite this stage before all warps finish reading it.
        cute.arch.sync_threads()
        read_stage += 1
        write_stage += 1
        if read_stage == STAGES:
            read_stage = 0
        if write_stage == STAGES:
            write_stage = 0

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # RMEM -> SMEM: rearrange lane-owned accumulators for contiguous stores.
    tCrD = cute.make_fragment_like(tCrC, cutlass.BFloat16)
    tCrD.store(tCrC.load().to(cutlass.BFloat16))
    cute.autovec_copy(tCrD, tCsC)
    cute.arch.sync_threads()

    # SMEM -> RMEM -> GMEM: each copy moves a contiguous 16-byte packet.
    thr_copy_C = c_tiled_copy.get_slice(tid)
    tCsC_epilogue = thr_copy_C.partition_S(sC)
    tCgC_epilogue = thr_copy_C.partition_D(gC)
    tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
    cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)
    cute.copy(c_tiled_copy, tCrC_epilogue, tCgC_epilogue)


@cute.jit
def multistage_gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        MMA_SHAPE_MNK,
    )
    tiled_mma = cute.make_tiled_mma(op, atom_layout_mnk=ATOM_LAYOUT_MNK)
    sA_layout = make_ab_smem_layout(CTA_M)
    sB_layout = make_ab_smem_layout(CTA_N)
    sC_layout = make_c_smem_layout()

    g2s_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.BFloat16,
        num_bits_per_copy=COPY_BITS,
    )
    c_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=COPY_BITS,
    )
    gmem_tiled_copy = make_tiled_copy(g2s_atom, CTA_K)
    c_tiled_copy = make_tiled_copy(c_atom, CTA_N)

    smem_bytes = cute.size_in_bytes(cutlass.BFloat16, sA_layout)
    smem_bytes += cute.size_in_bytes(cutlass.BFloat16, sB_layout)
    smem_bytes += cute.size_in_bytes(cutlass.BFloat16, sC_layout)

    multistage_gemm_kernel(
        a,
        b,
        c,
        sA_layout,
        sB_layout,
        sC_layout,
        gmem_tiled_copy,
        c_tiled_copy,
        tiled_mma,
    ).launch(
        grid=(
            cute.ceil_div(cute.size(c, mode=[0]), CTA_M),
            cute.ceil_div(cute.size(c, mode=[1]), CTA_N),
            1,
        ),
        block=(THREADS, 1, 1),
        smem=smem_bytes,
    )


def as_cute_tensor(tensor: torch.Tensor, divisibility: int) -> cute.Tensor:
    return (
        from_dlpack(tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=divisibility)
    )


def main(m: int, n: int, k: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if m % CTA_M or n % CTA_N or k % CTA_K:
        raise ValueError("M, N, and K must be multiples of 64, 64, and 32")
    if k < CTA_K * (STAGES - 1):
        raise ValueError("K must contain at least two 32-wide tiles")

    torch.manual_seed(2026)
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    a_cute = as_cute_tensor(a, k)
    b_cute = as_cute_tensor(b, k)
    out_cute = as_cute_tensor(out, n)

    compiled = cute.compile(multistage_gemm, a_cute, b_cute, out_cute)
    compiled(a_cute, b_cute, out_cute)
    torch.cuda.synchronize()

    reference = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    print(f"PASS: 3-stage BF16 GEMM ({m}, {n}, {k}), FP32 accumulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    args = parser.parse_args()
    main(args.m, args.n, args.k)
