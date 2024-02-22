import time
import numpy as np
cimport numpy as cnp

def stream(unsigned int STREAM_ARRAY_SIZE):
    cdef cnp.float64_t[:] a, b, c
    cdef double scalar

    a = np.ones(STREAM_ARRAY_SIZE, dtype=np.float64)
    b = np.ones(STREAM_ARRAY_SIZE, dtype=np.float64) * 2.0
    c = np.zeros(STREAM_ARRAY_SIZE, dtype=np.float64)
    scalar = 2.0

    times = [0, 0, 0, 0]
    timer = time.time_ns

    def copy():
        cdef unsigned int i
        times[0] = timer()
        for i in range(STREAM_ARRAY_SIZE):
            c[i] = a[i]
        times[0] = timer() - times[0]

    def scale():
        cdef unsigned int i
        times[1] = timer()
        for i in range(STREAM_ARRAY_SIZE):
            b[i] = scalar*c[i]
        times[1] = timer() - times[1]

    def add():
        cdef unsigned int i
        times[2] = timer()
        for i in range(STREAM_ARRAY_SIZE):
            c[i] = a[i]+b[i]
        times[2] = timer() - times[2]

    def triad():
        cdef unsigned int i
        times[3] = timer()
        for i in range(STREAM_ARRAY_SIZE):
            a[i] = b[i]+scalar*c[i]
        times[3] = timer() - times[3]

    copy()
    scale()
    add()
    triad()

    # Times are in ns, so without conversion, the calculation would be in GB/s
    return times