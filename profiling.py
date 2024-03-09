import argparse
from cythonized import run_cython
from cupy_code import run_cupy
from normal import run_normal
from numba_code import run_numba
from profilers import benchmark, benchmark_jit, duration_dict
from torch_code import run_torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", type=int, help="Number of particles", default=400)
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="enable plotting of the animation in realtime",
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
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Run the normal version of the code",
    )
    parser.add_argument(
        "--experiments",
        action="store",
        type=int,
        help="How many experiments to run for the average time",
        default=10,
    )

    args = parser.parse_args()

    if args.cython:
        benchmark(
            run_cython,
            args.experiments,
            args,
        )
    if args.numba:
        benchmark_jit(
            run_numba,
            args.experiments,
            args,
        )
    if args.cupy:
        benchmark(
            run_cupy,
            args.experiments,
            args,
        )
    if args.torch:
        benchmark(
            run_torch,
            args.experiments,
            args,
        )
    if args.normal:
        benchmark(
            run_normal,
            args.experiments,
            args,
        )

    for key, value in duration_dict.items():
        print(f"{key}: {value:.3f}s")


if __name__ == "__main__":
    main()
