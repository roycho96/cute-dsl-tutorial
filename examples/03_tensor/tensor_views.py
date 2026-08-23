import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def print_views_kernel(a: cute.Tensor):
    row = a[(1, None)]
    column = a[(None, 2)]
    tile = cute.local_tile(a, (2, 3), (1, 0))

    cute.printf("row A[1, None]")
    cute.print_tensor(row)
    cute.printf("column A[None, 2]")
    cute.print_tensor(column)
    cute.printf("tile local_tile(A, (2, 3), (1, 0))")
    cute.print_tensor(tile)


@cute.jit
def inspect_tensor(a: cute.Tensor):
    row = a[(1, None)]
    column = a[(None, 2)]
    tiled = cute.zipped_divide(a, (2, 3))
    tile = cute.local_tile(a, (2, 3), (1, 0))

    print("A layout:     ", a.layout)
    print("row layout:   ", row.layout)
    print("column layout:", column.layout)
    print("tiled layout: ", tiled.layout)
    print("tile layout:  ", tile.layout)

    print_views_kernel(a).launch(grid=(1, 1, 1), block=(1, 1, 1))


def main() -> None:
    torch_tensor = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 6)
    cute_tensor = from_dlpack(torch_tensor, assumed_align=16)

    compiled = cute.compile(inspect_tensor, cute_tensor)
    compiled(cute_tensor)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
