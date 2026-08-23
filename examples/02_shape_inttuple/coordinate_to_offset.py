import argparse

import cutlass
from cutlass import cute


@cute.jit
def inspect_layouts(row: cutlass.Int32, col: cutlass.Int32):
    shape = (2, 3)
    row_major = cute.make_layout(shape, stride=(3, 1))
    column_major = cute.make_layout(shape, stride=(1, 2))
    padded = cute.make_layout(shape, stride=(4, 1))
    default = cute.make_layout(shape)
    ordered_row = cute.make_ordered_layout(shape, order=(1, 0))
    ordered_column = cute.make_ordered_layout(shape, order=(0, 1))

    assert cute.size(row_major) == 6
    assert cute.cosize(row_major) == 6
    assert cute.cosize(column_major) == 6
    assert cute.cosize(padded) == 7
    assert cute.crd2idx((1, 2), row_major) == 5
    assert cute.crd2idx((1, 2), column_major) == 5
    assert default.stride == (1, 2)
    assert ordered_row.stride == row_major.stride
    assert ordered_column.stride == column_major.stride

    print("layouts")
    print("  row-major   ", row_major)
    print("  column-major", column_major)
    print("  padded      ", padded)
    print("  default     ", default)

    print("row-major offsets")
    print(
        "  ",
        cute.crd2idx((0, 0), row_major),
        cute.crd2idx((0, 1), row_major),
        cute.crd2idx((0, 2), row_major),
    )
    print(
        "  ",
        cute.crd2idx((1, 0), row_major),
        cute.crd2idx((1, 1), row_major),
        cute.crd2idx((1, 2), row_major),
    )

    print("column-major offsets")
    print(
        "  ",
        cute.crd2idx((0, 0), column_major),
        cute.crd2idx((0, 1), column_major),
        cute.crd2idx((0, 2), column_major),
    )
    print(
        "  ",
        cute.crd2idx((1, 0), column_major),
        cute.crd2idx((1, 1), column_major),
        cute.crd2idx((1, 2), column_major),
    )

    print("size / cosize")
    print("  row-major", cute.size(row_major), cute.cosize(row_major))
    print("  padded   ", cute.size(padded), cute.cosize(padded))

    cute.printf("coordinate: ({}, {})", row, col)
    cute.printf("row-major offset: {}", cute.crd2idx((row, col), row_major))
    cute.printf("column-major offset: {}", cute.crd2idx((row, col), column_major))
    cute.printf("padded offset: {}", cute.crd2idx((row, col), padded))


def main(row: int, col: int) -> None:
    if not 0 <= row < 2 or not 0 <= col < 3:
        raise ValueError("coordinate must be inside Shape (2, 3)")

    compiled = cute.compile(
        inspect_layouts,
        cutlass.Int32(row),
        cutlass.Int32(col),
    )
    compiled(cutlass.Int32(row), cutlass.Int32(col))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--col", type=int, default=1)
    args = parser.parse_args()
    main(args.row, args.col)
