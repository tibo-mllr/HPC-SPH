# Final assignment for Introduction to High Performance Computing

## Create Your Own Smoothed-Particle Hydrodynamics Simulation

Authors: EBENAU Luuk, KHERA Palak, MULLER Thibault Kungliga Tekniska Högskolan (KTH), 2024

> Original project and code from Philip Mocz (2020) Princeton Univeristy, [@PMocz](https://twitter.com/PMocz)<br> > [📝 Read the Algorithm Write-up on Medium](https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1)

## Introduction

In the final assignment for the course "Introduction to High Performance Computing" at KTH, we did research in improving the original algorithm by Philip Mocz.
The original project aims to simulate stellar phenomena such as star formation using the Smoothed-Particle-Hydrodynamics (SPH) model, which represents fluids as a collection of interacting particles. The computation includes the Euler equation of an ideal fluid, and uses properties such as mass, position, velocity, gravity and viscosity. A Gaussian smoothing kernel is used to distribute the particles in space, aiming to study and visualize complex fluid behavior in a stellar context.

## Improvement methods

1. Cython
2. Numba (jit)
3. CuPy (gpu)
4. PyTorch (gpu)

## System requirements

The code has been developed to be run on Linux. Using a different operating system might result in unexpected behaviour. For running the code on windows, it is recommended to use WSL2 instead.

Additionally, the optimisation using CuPy and PyTorch make use of the GPU. For these codes to work a working Cuda installation matching the pytorch/cupy version has to be present.

## Installation

To install all the required packages, make sure you have python installed. Then

```bash
pip3 install -r requirements.txt
```

For the installation of gpu, the guide at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) can be followed for PyTorch.

## Usage

## Sphinx documentation

The documentation of the code can be found in the [docs](./docs) folder. To generate the documentation, the following command can be used:

```bash
cd docs
make html
```

You can then access the documentation by opening the `index.html` file in the `_build/html` folder.

### Setup

First, you have to comply the cython code using the following command:

```bash
python3 cythonized/setup.py build_ext --inplace
```

### Running the code

Using [run.py](./run.py) the different methods can be run.

You can add all these of parameters:

- `-N` to set the number of particles, default 400
- `--plot` to enable or disable the plotting of the results, default False
- `--realTime` to enable or disable the plotting of the results in real time, default False (if `plot` is disabled, `realTime` is disabled whatever the value is)
- `--cython` to enable or disable the cython implementation
- `--numba` to enable or disable the numba implementation
- `--cupy` to enable or disable the cupy implementation
- `--torch` to enable or disable the pytorch implementation
- `--normal` to enable or disable the normal implementation

For example, to run the PyTorch implementation with 1000 particles and plot in real time:

```bash
python3 run.py -N 1000 --plot --realTime --torch
```

### Benchmarking speed

Using [time_profiling.py](./time_profiling.py) the speed of the different methods can be compared.

You can add all these of parameters:

- `-N` to set the number of particles, default 400
- `--plot` to enable or disable the plotting of the results, default False
- `--cython` to enable or disable the cython implementation
- `--numba` to enable or disable the numba implementation
- `--cupy` to enable or disable the cupy implementation
- `--torch` to enable or disable the pytorch implementation
- `--normal` to enable or disable the normal implementation
- `--experiments` to chose the number of experiments to make the average onThe code can be run using, default 10

For example, to make the benchmark on 5 experiments of all the methods with 1000 particles and not plot the results:

```bash
python3 time_profiling.py -N 1000 --cython --numba --cupy --torch --normal --experiments 5
```

### Benchmarking methods over the number of particles

Using [benchmark.py](./benchmark.py) the speed of the different methods can be compared over the number of particles.

You can add all these of parameters:

- `minimum` to set the minimum number of particles, default 100
- `maximum` to set the maximum number of particles, default 1000
- `step` to set the step between the number of particles, default 100
- `--plot` to enable or disable the plotting of the results (not those of the benchmark, but the figure of the particle), default False
- `--cython` to enable or disable the cython implementation
- `--numba` to enable or disable the numba implementation
- `--cupy` to enable or disable the cupy implementation
- `--torch` to enable or disable the pytorch implementation
- `--normal` to enable or disable the normal implementation
- `--experiments` to chose the number of experiments to make the average onThe code can be run using, default 10

For example, to make the benchmark on 5 experiments of all the methods with 100 to 1000 particles with a step of 200:

```bash
python3 benchmark.py --minimum 400 --maximum 2000 --step 200 --cython --numba --cupy --torch --normal --experiments 5
```

![Simulation](./sph.png)

# Tests

We made unit tests to make sure that the algorithm still has the same outputs after our optimization method. We did this by comparing the outputs of the original algorithm with that of the new code.

