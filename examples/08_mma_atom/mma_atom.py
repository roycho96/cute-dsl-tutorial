import cutlass
from cutlass import cute

MMA_SHAPE_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 2, 1)


@cute.jit
def inspect_mma():
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        MMA_SHAPE_MNK,
    )
    tiled_mma = cute.make_tiled_mma(
        op,
        atom_layout_mnk=ATOM_LAYOUT_MNK,
    )

    print("MMA operation")
    print(cute.pretty_str(op))
    print("\nTiledMMA")
    print(cute.pretty_str(tiled_mma))
    print("\nTiled A layout: thread/value -> (M, K)")
    print(cute.pretty_str(tiled_mma.tv_layout_A_tiled))
    print("\nTiled B layout: thread/value -> (N, K)")
    print(cute.pretty_str(tiled_mma.tv_layout_B_tiled))
    print("\nTiled C layout: thread/value -> (M, N)")
    print(cute.pretty_str(tiled_mma.tv_layout_C_tiled))
    print("\nTiled MMA shape")
    print(
        tiled_mma.get_tile_size(0),
        tiled_mma.get_tile_size(1),
        tiled_mma.get_tile_size(2),
    )


def main() -> None:
    compiled = cute.compile(inspect_mma)
    compiled()


if __name__ == "__main__":
    main()
