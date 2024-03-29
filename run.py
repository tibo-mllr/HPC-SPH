import argparse
from cythonized import run_cython
from cupy_code import run_cupy
from torch_code import run_torch
from normal import run_normal
from numba_code import run_numba


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", type=int, help="Number of particles", default=400)
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Enable plotting of the animation",
    )
    parser.add_argument(
        "--realTime",
        default=False,
        action="store_true",
        help="Enable plotting of the animation in realtime",
    )
    parser.add_argument(
        "--cython",
        action="store_true",
        help="Run the cythonized version of the code",
    )
    parser.add_argument(
        "--numba",
        action="store_true",
        help="Run the numba version of the code",
    )
    parser.add_argument(
        "--cupy",
        action="store_true",
        help="Run the GPU version of the code",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Run the GPU version of the code",
    )

    args = parser.parse_args()

    if args.cython:
        run_cython(args)
    elif args.numba:
        run_numba(args)
    elif args.cupy:
        run_cupy(args)
    elif args.torch:
        run_torch(args)
    else:
        run_normal(args)


if __name__ == "__main__":
    main()
