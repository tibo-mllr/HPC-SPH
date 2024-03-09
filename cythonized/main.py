import numpy as np
import matplotlib.pyplot as plt
from .sph_cython import getDensity, main
from scipy.special import gamma

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""


def run(args):
    """SPH simulation"""

    N = args.N
    plotRealTime = args.plot
    # Simulation parameters
    # N = 400  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 12  # time at which simulation ends
    dt = 0.04  # timestep
    M = 2  # star mass
    R = 0.75  # star radius
    h = 0.1  # smoothing length
    k = 0.1  # equation of state constant
    n = 1  # polytropic index
    nu = 1  # damping

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    lmbda = (
        2
        * k
        * (1 + n)
        * np.pi ** (-3 / (2 * n))
        * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n)
        / R**2
    )  # ~ 2.01
    m = M / N  # single particle mass
    pos = np.random.randn(N, 3)  # randomly selected positions and velocities

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, 1, 100)
    rr[:, 0] = rlin
    rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)

    pos = main(N, Nt, dt, m, h, k, n, lmbda, nu)

    # get density for plotting
    rho = np.asarray(getDensity(pos, pos, m, h))

    if args.plot:

        plt.sca(ax1)
        plt.cla()
        cval = np.minimum((rho - 3) / 3, 1).flatten()
        plt.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
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
        plt.plot(rlin, rho_analytic, color="gray", linewidth=2)
        rho_radial = np.asarray(getDensity(rr, pos, m, h))
        plt.plot(rlin, rho_radial, color="blue")
        plt.pause(0.001)

        # add labels/legend
        plt.sca(ax2)
        plt.xlabel("radius")
        plt.ylabel("density")

        plt.show()

    return 0
