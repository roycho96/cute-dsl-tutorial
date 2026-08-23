import argparse

import cutlass
from cutlass import cute


@cute.jit
def inspect_shapes(m: cutlass.Int32, n: cutlass.Constexpr[int]):
    scalar = 6
    singleton = (6,)
    flat = (2, 3, 4)
    hierarchical = (2, (3, 4))
    mixed = (m, (3, n))

    assert cute.rank(hierarchical) == 2
    assert cute.rank(hierarchical, mode=[1]) == 2
    assert cute.depth(hierarchical) == 2
    assert cute.size(hierarchical) == 24
    assert cute.size(hierarchical, mode=[1]) == 12

    print("shape             rank depth size")
    print("  6             ", cute.rank(scalar), cute.depth(scalar), cute.size(scalar))
    print(
        "  (6,)          ",
        cute.rank(singleton),
        cute.depth(singleton),
        cute.size(singleton),
    )
    print("  (2, 3, 4)     ", cute.rank(flat), cute.depth(flat), cute.size(flat))
    print(
        "  (2, (3, 4))  ",
        cute.rank(hierarchical),
        cute.depth(hierarchical),
        cute.size(hierarchical),
    )
    print("mode [1]       ", cute.get(hierarchical, mode=[1]))
    print("mode [1, 0]    ", cute.get(hierarchical, mode=[1, 0]))
    print("mixed at compile time:", mixed)

    cute.printf("mixed at runtime: {}", mixed)
    cute.printf("mixed size: {}", cute.size(mixed))


def main(m: int, n: int) -> None:
    if m <= 0 or n <= 0:
        raise ValueError("m과 n은 양수여야 합니다.")

    compiled = cute.compile(inspect_shapes, cutlass.Int32(m), n)
    compiled(cutlass.Int32(m))
    compiled(cutlass.Int32(m + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    main(args.m, args.n)
