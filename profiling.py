import argparse
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
        "--experiments",
        action="store",
        help="How many experiments to run for the average time",
        default=10,
    )

    args = parser.parse_args()

    for i in range(args.experiments):
        print(f"Experiment {i+1}/{args.experiments}")
        if args.cython:
            run_cython()
        elif args.numba:
            run_numba()
        # elif args.gpu:
        #     run_gpu()
        else:
            run_normal()

    for key, value in duration_dict.items():
        if "numba" in key:
            value = value[1:]
        print(
            f"Average time ({len(value)} exp) for {key}: {((sum(value)/len(value))/1e9):.3e} s"
        )
        print(
            f"Standard deviation for {key}: {((sum([(x - sum(value)/len(value))**2 for x in value])/len(value))/1e9):.3e} s"
        )


if __name__ == "__main__":
    main()
