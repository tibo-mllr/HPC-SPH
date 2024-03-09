# Final assignment for Introduction to High Performance Computing
## Create Your Own Smoothed-Particle Hydrodynamics Simulation
Authors: EBENAU Luuk, KHERA Palak, MULLER Thibault Kungliga Tekniska Högskolan (KTH), 2024

> Original project and code from Philip Mocz (2020) Princeton Univeristy, [@PMocz](https://twitter.com/PMocz)<br>
[📝 Read the Algorithm Write-up on Medium](https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1)

## Introduction
In the final assignment for the course "Introduction to High Performance Computing" at KTH, we did research in improving the original algorithm by Philip Mocz. 
The original project aims to simulate stellar phenomena such as star formation using the Smoothed-Particle-Hydrodynamics (SPH) model, which represents fluids as a collection of interacting particles. The computation includes the Euler equation of an ideal fluid, and uses properties such as mass, position, velocity, gravity and viscosity. A Gaussian smoothing kernel is used to distribute the particles in space, aiming to study and visualize complex fluid behavior in a stellar context.

## Inprovement methods
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

## Running code
### Benchmarking speed
Using [profiling.py](./profiling.py) the speed of the different methods can be compared.

You can add all these of parameters:
- `-N` to set the number of particles, default 400
- `--plot` to enable or disable the plotting of the results, default False
- `--cython` to enable or disable the cython implementation
- `--numba` to enable or disable the numba implementation
- `--cupy` to enable or disable the cupy implementation
- `--torch` to enable or disable the pytorch implementation
- `--normal` to enable or disable the normal implementation
- `--experiments` to chose the number of experiments to make the average onThe code can be run using, default 10
```bash
python3 profiling.py
```

For example:
```bash
python3 profiling.py -N 1000 --cython --numba --cupy --torch --normal --experiments 5
```
To make the benchmark on 5 experiments of all the methods with 1000 particles and not plot the results.


![Simulation](./sph.png)
