from cythonized import run_cython

# from gpu import run_gpu
from normal import run_normal
from numba_code import run_numba
from .tools import timefn


@timefn
def run_normal_profiling():
    run_normal(False)


@timefn
def run_cython_profiling():
    run_cython(False)


@timefn
def run_numba_profiling():
    run_numba(False)


# @timefn
# def run_gpu_profiling():
#     run_gpu(False)
