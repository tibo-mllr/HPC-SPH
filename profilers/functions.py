from .tools import timer, timefn


@timefn
def benchmark_jit(fn, i_count, args):
    total_time = 0
    first_run_time = 0
    for i in range(i_count):
        t1 = timer()
        fn(args)
        t2 = timer()
        time_passed = (t2 - t1) / 1e9
        if i == 0:
            first_run_time = time_passed
        else:
            total_time += time_passed

    return total_time / (i_count - 1), first_run_time


@timefn
def benchmark(fn, i_count, args):
    total_time = 0
    for _ in range(i_count):
        t1 = timer()
        fn(args)
        t2 = timer()
        time_passed = (t2 - t1) / 1e9
        total_time += time_passed

    return total_time / i_count
