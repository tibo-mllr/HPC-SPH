##### PROFILING CODE #####

from functools import wraps
import time
timer = time.time_ns # Best precision for my machine
serial_durations, pure_durations = [], []

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = timer()
        result = fn(*args, **kwargs)
        t2 = timer()

        # Append durations to the correct list, in order to compute the average and especially std
        if "serial" in fn.__name__:
            serial_durations.append((t2 - t1)/1e9)
        elif "pure" in fn.__name__:
            pure_durations.append((t2 - t1)/1e9)
        else: 
            print(f"{fn.__name__} took {(t2 - t1)/1e9:.4f} s")
            
        return result
    return measure_time

##### END PROFILING CODE #####

# import time
# time_before = time.time_ns()
# ...
# time_after = time.time_ns()

# total_time = (time_after - time_before) / 1e9
# print(f"the function took {total_time:.4f} s" )