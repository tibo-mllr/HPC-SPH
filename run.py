import argparse
from cythonized import run_cython
from gpu import run_gpu
from normal import run_normal
from numba_code import run_numba
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cython-setup",
        action="store_true",
        help="Run the cythonized setup.py file to compile the cython code",
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
        "--gpu",
        action="store_true",
        help="Run the GPU version of the code",
    )

    args = parser.parse_args()

    if args.cython_setup:
        subprocess.run(["python3", "cythonized/setup.py", "build_ext", "--inplace"])
    elif args.cython:
        run_cython()
    elif args.numba:
        run_numba()
    elif args.gpu:
        run_gpu()
    else:
        run_normal()


if __name__ == "__main__":
    main()
