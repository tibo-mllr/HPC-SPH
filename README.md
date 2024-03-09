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
Inside the benchmarking folder, first run ```python setup.py build_ext --inplace```. Next, the speeds of the different algorithm can be benchmarked by running the code inside comparing_algorithms.ipynb

### Attempts to improve the performance

#### 1. Cython

Cython is a superset of Python that allows you to write C extensions for Python. It is a language that makes writing C extensions for Python as easy as Python itself. Cython is a compiler that translates Python-like code files into C code. This C code can then be compiled into a Python extension module.

Two cython versions of the gradW function have been made. However, this has been counterproductive as the performance of the code has decreased. The reason for this is that the overhead of calling the C function is too large compared to the time saved by the C function, due to the huge times the function is called. We will try to cythonize also the main loop of the code.



![Simulation](./sph.png)
