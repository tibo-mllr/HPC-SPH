from .functions import (
    run_cython_profiling as run_cython,
    # run_gpu_profiling as run_gpu,
    run_normal_profiling as run_normal,
    run_numba_profiling as run_numba,
)
from .tools import duration_dict
