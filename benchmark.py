import argparse
from cythonized import run_cython
from cupy_code import run_cupy
from matplotlib import pyplot as plt
from normal import run_normal
from numba_code import run_numba
from profilers import benchmark, benchmark_jit
from torch_code import run_torch


def main():
    parser = argparse.ArgumentParser()
    time_normal, time_cython, time_numba, time_cupy, time_torch = [], [], [], [], []

    parser.add_argument(
        "--minimum",
        action="store",
        type=int,
        help="Minimum number of particles",
        default=100,
    )
    parser.add_argument(
        "--maximum",
        action="store",
        type=int,
        help="Maximum number of particles",
        default=1000,
    )
    parser.add_argument(
        "--step",
        action="store",
        type=int,
        help="Increment of particles",
        default=100,
    )
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
    args.realTime = False

    for N in range(args.minimum, args.maximum + 1, args.step):
        args.N = N
        if args.cython:
            avg_time_cython = benchmark(
                run_cython,
                args.experiments,
                args,
            )
            time_cython.append(avg_time_cython)
        if args.numba:
            avg_time_numba, first_run = benchmark_jit(
                run_numba,
                args.experiments,
                args,
            )
            time_numba.append(avg_time_numba)
        if args.cupy:
            avg_time_cupy = benchmark(
                run_cupy,
                args.experiments,
                args,
            )
            time_cupy.append(avg_time_cupy)
        if args.torch:
            avg_time_torch = benchmark(
                run_torch,
                args.experiments,
                args,
            )
            time_torch.append(avg_time_torch)
        if args.normal:
            avg_time_normal = benchmark(
                run_normal,
                args.experiments,
                args,
            )
            time_normal.append(avg_time_normal)

    times_to_plot = [
        lst
        for lst in [time_normal, time_cython, time_numba, time_cupy, time_torch]
        if lst
    ]
    number_of_plots = len(times_to_plot)

    fig, ax = plt.subplots(number_of_plots, 1, figsize=(10, 10))
    fig.suptitle("Comparison of different implementations")
    for i, times in enumerate(times_to_plot):
        ax[i].plot(range(args.minimum, args.maximum + 1, args.increment), times)
        ax[i].set_xlabel("Number of particles")
        ax[i].set_ylabel("Average time (s)")
        ax[i].set_title(
            [
                "Normal",
                "Cython",
                "Numba",
                "Cupy",
                "Torch",
            ][i]
        )

    plt.show()


if __name__ == "__main__":
    main()
