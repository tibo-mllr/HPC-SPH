from functools import wraps
from time import time_ns

timer = time_ns
duration_dict = {}


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        result = fn(*args, **kwargs)

        if isinstance(result, tuple):
            avg_time, first_time_run = result
            duration_dict["numba_run"] = avg_time
            duration_dict["numba_first_run"] = first_time_run
        else:
            duration_dict[args[0].__name__] = result

        return result

    return measure_time
