import argparse
from math import sqrt
from profilers import run_cython, run_normal, run_numba, duration_dict


def main():
    parser = argparse.ArgumentParser()
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
        "--gpu",
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

    for i in range(args.experiments):
        print(f"Experiment {i+1}/{args.experiments}")
        if args.cython:
            run_cython()
        if args.numba:
            run_numba()
        # if args.gpu:
        #     run_gpu()
        if args.normal:
            run_normal()

    for key, value in duration_dict.items():
        if "numba" in key:
            value = value[1:]

        # Calculate average
        avg = sum(value) / len(value)

        # Calculate standard deviation
        variance = sum((x - avg) ** 2 for x in value) / len(value)
        std_dev = sqrt(variance)

        print(f"Average time ({len(value)} exp) for {key}: {(avg / 1e9):.3e} s")
        print(f"Standard deviation for {key}: {(std_dev / 1e9):.3e} s")
        print()


if __name__ == "__main__":
    main()
