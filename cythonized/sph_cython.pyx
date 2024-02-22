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

def getPairwiseSeparations(cnp.ndarray ri, cnp.ndarray rj):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    cdef unsigned int M = ri.shape[0]
    cdef unsigned int N = rj.shape[0]

    # positions ri = (x,y,z)
    cdef cnp.ndarray rix = ri[:, 0].reshape((M, 1))
    cdef cnp.ndarray riy = ri[:, 1].reshape((M, 1))
    cdef cnp.ndarray riz = ri[:, 2].reshape((M, 1))

    # other set of points positions rj = (x,y,z)
    cdef cnp.ndarray rjx = rj[:, 0].reshape((N, 1))
    cdef cnp.ndarray rjy = rj[:, 1].reshape((N, 1))
    cdef cnp.ndarray rjz = rj[:, 2].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    cdef cnp.float64_t[:, :] dx = rix - rjx.T
    cdef cnp.float64_t[:, :] dy = riy - rjy.T
    cdef cnp.float64_t[:, :] dz = riz - rjz.T

    return dx, dy, dz

def W(cnp.float64_t[:, :] x, cnp.float64_t[:, :] y, cnp.float64_t[:, :] z, float h):
    """
    Gaussian Smoothing kernel (3D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        w     is the evaluated smoothing function
    """

    cdef cnp.ndarray r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

    cdef cnp.ndarray w = (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)

    return w

def getDensity(cnp.ndarray r, cnp.ndarray pos, float m, float h):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    cdef unsigned int M = r.shape[0]

    cdef tuple[cnp.ndarray[cnp.float64_t], cnp.ndarray[cnp.float64_t], cnp.ndarray[cnp.float64_t]] result = getPairwiseSeparations(r, pos)

    cdef cnp.float64_t[:, :] dx = result[0], dy = result[1], dz = result[2]
    cdef cnp.float64_t[:, :] rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))

    return rho

def getPressure(cnp.float64_t[:, :] rho, float k, unsigned int n):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """

    cdef cnp.float64_t[:, :] P = k * np.power(rho, (1 + 1 / n))

    return P

def getAcc(cnp.ndarray pos, cnp.ndarray vel, float m, float h, float k, unsigned int n, float lmbda, unsigned int nu):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """

    cdef unsigned int N = pos.shape[0]

    # Calculate densities at the position of the particles
    cdef cnp.float64_t[:, :] rho_mem = getDensity(pos, pos, m, h)
    cdef cnp.ndarray rho = np.asarray(rho_mem)

    # Get the pressures
    cdef cnp.float64_t[:, :] P = getPressure(rho, k, n)

    # Get pairwise distances and gradients
    cdef tuple[cnp.ndarray[cnp.float64_t], cnp.ndarray[cnp.float64_t], cnp.ndarray[cnp.float64_t]] result = getPairwiseSeparations(pos, pos)
    cdef cnp.float64_t[:, :] dx = result[0], dy = result[1], dz = result[2]

    result = gradW1(dx, dy, dz, h)
    cdef cnp.float64_t[:, :] dWx = result[0], dWy = result[1], dWz = result[2]

    # Add Pressure contribution to accelerations
    cdef cnp.float64_t[:, :] ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, 1).reshape((N, 1))
    cdef cnp.float64_t[:, :] ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, 1).reshape((N, 1))
    cdef cnp.float64_t[:, :] az = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, 1).reshape((N, 1))

    # pack together the acceleration components
    cdef cnp.ndarray a = np.hstack((ax, ay, az))

    # Add external potential force
    a -= lmbda * pos

    # Add viscosity
    a -= nu * vel

    return a

def main(unsigned int N, unsigned int Nt, float dt, float m, float h, float k, unsigned int n, float lmbda, unsigned int nu):
    cdef unsigned int i
    cdef cnp.ndarray pos = np.random.randn(N, 3)
    cdef cnp.ndarray vel = np.zeros((N, 3))
    cdef cnp.ndarray acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2

        # drift
        pos += vel * dt

        # update accelerations
        acc = np.asarray(getAcc(pos, vel, m, h, k, n, lmbda, nu))

        # (1/2) kick
        vel += acc * dt / 2
    
    return pos