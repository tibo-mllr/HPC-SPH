from functools import wraps
from time import time_ns

timer = time_ns
duration_dict = {}


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()

        # Append durations to the correct list, in order to compute the average and especially std
        if fn.__name__ in duration_dict:
            duration_dict[fn.__name__].append(t2 - t1)
        else:
            duration_dict[fn.__name__] = [t2 - t1]
        print(f"Execution of {fn.__name__} took {((t2 - t1)/1e9):.3e} s")

        return result

    return measure_time
