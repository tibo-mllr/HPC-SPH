import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""


def W(x, y, z, h):
    """
    Gaussian Smoothing kernel (3D) using CuPy for GPU acceleration
        x     is a vector/matrix of x positions, converted to CuPy array
        y     is a vector/matrix of y positions, converted to CuPy array
        z     is a vector/matrix of z positions, converted to CuPy array
        h     is the smoothing length (scalar or CuPy array)
        w     is the evaluated smoothing function
    """

    val = x**2 + y**2 + z**2
    r = cp.sqrt(val)

    w = (1.0 / (h * cp.sqrt(cp.pi))) ** 3 * cp.exp(-(r**2) / h**2)

    return w


def gradW(x, y, z, h):
    """
    Gradient of the Gaussian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """

    r = cp.sqrt(x**2 + y**2 + z**2)

    n = -2 * cp.exp(-(r**2) / h**2) / h**5 / (cp.pi) ** (3 / 2)
    wx = n * x
    wy = n * y
    wz = n * z

    return wx, wy, wz


def getPairwiseSeparations(ri, rj):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))

    # other set of points positions rj = (x,y,z)
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T

    return dx, dy, dz


def getDensity(r, pos, m, h):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparations(r, pos)

    rho = cp.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))

    return rho


def getPressure(rho, k, n):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """

    P = k * rho ** (1 + 1 / n)

    return P


def getAcc(pos, vel, m, h, k, n, lmbda, nu):
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

    N = pos.shape[0]

    # Calculate densities at the position of the particles
    rho = getDensity(pos, pos, m, h)

    # Get the pressures
    P = getPressure(rho, k, n)

    # Get pairwise distances and gradients
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)

    # Add Pressure contribution to accelerations
    ax = -cp.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -cp.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, 1).reshape((N, 1))
    az = -cp.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, 1).reshape((N, 1))

    # pack together the acceleration components
    a = cp.hstack((ax, ay, az))

    # Add external potential force
    a -= lmbda * pos

    # Add viscosity
    a -= nu * vel

    return a


def run_cupy(args):
    """SPH simulation"""

    # Simulation parameters
    N = 400  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 12  # time at which simulation ends
    dt = 0.04  # timestep
    M = 2  # star mass
    R = 0.75  # star radius
    h = 0.1  # smoothing length
    k = 0.1  # equation of state constant
    n = 1  # polytropic index
    nu = 1  # damping
    plotRealTime = args.realTime  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    cp.random.seed(42)  # set the random number generator seed

    lmbda = (
        2
        * k
        * (1 + n)
        * cp.pi ** (-3 / (2 * n))
        * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n)
        / R**2
    )  # ~ 2.01

    m = M / N  # single particle mass
    pos = cp.random.randn(N, 3)  # randomly selected positions and velocities
    vel = cp.zeros(pos.shape)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
    # number of timesteps
    Nt = int(cp.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    rr = cp.zeros((100, 3))
    rlin = cp.linspace(0, 1, 100)
    rr[:, 0] = rlin
    rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)

        # (1/2) kick
        vel += acc * dt / 2

        # update time
        t += dt

        # get density for plotting
        rho = getDensity(pos, pos, m, h)

        # plot in real time
        if args.plot and (plotRealTime or (i == Nt - 1)):
            plt.sca(ax1)
            plt.cla()

            # Ensure cval is a NumPy array
            cval = np.minimum(
                (rho.get() - 3) / 3, 1
            ).flatten()  # Assuming rho is a CuPy array

            # Convert pos1 to a NumPy array
            pos1 = pos.get()

            plt.scatter(
                pos1[:, 0], pos1[:, 1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5
            )
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect("equal", "box")
            ax1.set_xticks([-1, 0, 1])
            ax1.set_yticks([-1, 0, 1])
            ax1.set_facecolor("black")
            ax1.set_facecolor((0.1, 0.1, 0.1))

            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, 1), ylim=(0, 3))
            ax2.set_aspect(0.1)

            # Assuming rlin and rho_analytic are already NumPy arrays or compatible types
            rlin_np = rlin.get()
            plt.plot(rlin_np, rho_analytic.get(), color="gray", linewidth=2)

            # Ensure rho_radial is a NumPy array
            rho_radial = getDensity(
                rr, pos, m, h
            ).get()  # Assuming the result is a CuPy array

            plt.plot(rlin_np, rho_radial, color="blue")
            plt.pause(0.001)

    if args.plot:
        # add labels/legend
        plt.sca(ax2)
        plt.xlabel("radius")
        plt.ylabel("density")

        plt.show()

    return 0
