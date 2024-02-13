import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, exp, pow, M_PI

def gradW1(cnp.float64_t[:, :] x, cnp.float64_t[:, :] y, cnp.float64_t[:, :] z, float h):
    """
    Gradient of the Gaussian Smoothing kernel (3D)
    x     is a matrix of x positions
    y     is a matrix of y positions
    z     is a matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """
    cdef cnp.float64_t[:, :] wx, wy, wz
    cdef cnp.float64_t[:, :] r
    cdef cnp.float64_t[:, :] n

    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

    n = -2 * np.exp(-np.power(r, 2) / np.power(h, 2)) / np.power(h, 5) / (np.pi) ** (3 / 2)
    wx = np.multiply(n, x)
    wy = np.multiply(n, y)
    wz = np.multiply(n, z)

    return wx, wy, wz

def gradW2(cnp.float64_t[:, :] x, cnp.float64_t[:, :] y, cnp.float64_t[:, :] z, float h):
   """
   Gradient of the Gaussian Smoothing kernel (3D)
   x     is a matrix of x positions
   y     is a matrix of y positions
   z     is a matrix of z positions
   h     is the smoothing length
   wx, wy, wz     is the evaluated gradient
   """
   cdef cnp.float64_t[:, :] wx = np.empty_like(x), wy = np.empty_like(y), wz = np.empty_like(z)
   cdef cnp.float64_t[:, :] r = np.empty_like(x)
   cdef cnp.float64_t[:, :] n = np.empty_like(x)
   cdef Py_ssize_t i, j


   for i in range(x.shape[0]):
       for j in range(x.shape[1]):
           r[i, j] = sqrt(pow(x[i, j], 2) + pow(y[i, j], 2) + pow(z[i, j], 2))
           n[i, j] = -2 * exp(-pow(r[i, j], 2) / pow(h, 2)) / pow(h, 5) / pow(M_PI, 3 / 2)
           wx[i, j] = n[i, j] * x[i, j]
           wy[i, j] = n[i, j] * y[i, j]
           wz[i, j] = n[i, j] * z[i, j]


   return wx, wy, wz