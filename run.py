import argparse
from cythonized import run_cython
from normal import run_normal
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cython",
        action="store_true",
        help="Run the cythonized version of the code",
    )
    parser.add_argument(
        "--cython-setup",
        action="store_true",
        help="Run the cythonized setup.py file to compile the cython code",
    )

    args = parser.parse_args()

    if args.cython_setup:
        subprocess.run(["python3", "cythonized/setup.py", "build_ext", "--inplace"])
    elif args.cython:
        run_cython()
    else:
        run_normal()


if __name__ == "__main__":
    main()
