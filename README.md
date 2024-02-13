# sph-python

Smoothed-Particle Hydrodynamics simulation of Toy Star

## Create Your Own Smoothed-Particle Hydrodynamics Simulation (With Python)

### EBENAU Luuk, KHERA Palak, MULLER Thibault Kungliga Tekniska Högskolan (KTH), 2024

> Original project and code from Philip Mocz (2020) Princeton Univeristy, [@PMocz](https://twitter.com/PMocz)

### [📝 Read the Algorithm Write-up on Medium](https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1)

Simulate a toy star with SPH

```bash
pip3 install -r requirements.txt
python3 sph.py
```

![Simulation](./sph.png)

### Attempts to improve the performance

#### 1. Cython

Cython is a superset of Python that allows you to write C extensions for Python. It is a language that makes writing C extensions for Python as easy as Python itself. Cython is a compiler that translates Python-like code files into C code. This C code can then be compiled into a Python extension module.

Two cython versions of the gradW function have been made. However, this has been counterproductive as the performance of the code has decreased. The reason for this is that the overhead of calling the C function is too large compared to the time saved by the C function, due to the huge times the function is called. We will try to cythonize also the main loop of the code.