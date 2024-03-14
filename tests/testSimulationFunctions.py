import sys
from testconfig import root_folder
sys.path.append(root_folder)
import pytest
# now the root modules should be available. Make sure you change the root folder to the right path

import normal
import numba_code
import cythonized
import torch_code
import cupy_code 
import numpy as np
from scipy.special import gamma
import cupy as cp
import torch


class Arg:
	def __init__(self, N=400, plot=False):
		self.N = N 
		self.plot = plot
args = Arg(N=100,plot=False)

@pytest.mark.parametrize('acc1, acc2',[(normal.getAcc, cupy_code.getAcc)])
def test_acc_cupy(acc1,acc2):
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
	plotRealTime = False  # switch on for plotting as the simulation goes along

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

	acc = cupy_code.getAcc(pos, vel, m, h, k, n, lmbda, nu)

	test_accuracy_function(acc)

@pytest.mark.parametrize('acc1, acc2',[(normal.getAcc, numba_code.getAcc)])
def test_acc_numba(acc1,acc2):
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
	# plotRealTime = True  # switch on for plotting as the simulation goes along

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
	vel = np.zeros(pos.shape)

	# calculate initial gravitational accelerations************************
	# time_before = time.time_ns()
	acc = numba_code.getAcc(pos, vel, m, h, k, n, lmbda, nu)

	test_accuracy_function(acc)

# @pytest.mark.parametrize('acc1, acc2',[(normal.getAcc, cythonized.getAcc)])
# def test_acc_cython(acc1,acc2):
# 	test_accuracy_function(acc1,acc2)

@pytest.mark.parametrize('acc1, acc2',[(normal.getAcc, torch_code.getAcc)])
def test_acc_torch(acc1,acc2):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	pi_tensor = torch.tensor(torch.pi, device=device)
	# Simulation parameters
	N = args.N
	plotRealTime = args.plot
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
	# plotRealTime = True  # switch on for plotting as the simulation goes along

	# Generate Initial Conditions
	# torch.random.seed()  # set the random number generator seed
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	
	lmbda = (
			2
			* k
			* (1 + n)
			* np.pi ** (-3 / (2 * n))
			* (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n)
			/ R**2
	)  # ~ 2.01

	m = M / N  # single particle mass
	
	lmbda = torch.tensor(lmbda, dtype=torch.float32, device=device)
	# N = torch.tensor(N, dtype=torch.float32, device=device)
	m = torch.tensor(m, dtype=torch.float32, device=device)
	h = torch.tensor(h, dtype=torch.float32, device=device)
	k = torch.tensor(k, dtype=torch.float32, device=device)
	n = torch.tensor(n, dtype=torch.float32, device=device)
	nu = torch.tensor(nu, dtype=torch.float32, device=device)

	pos = torch.randn(N,3,device=device)  # randomly selected positions and velocities
	vel = torch.zeros(pos.shape,device=device)

	# calculate initial gravitational accelerations
	acc = torch_code.getAcc(pos, vel, m, h, k, n, lmbda, nu)

	test_accuracy_function(acc)


def getNormalAccuracy():

	N = args.N
	plotRealTime = args.plot
	t = 0 
	tEnd = 12
	dt = 0.04
	M = 2
	R = 0.75
	h = 0.1
	k = 0.1
	n = 1
	nu = 1 

	np.random.seed(42)

	lmbda = (
			2
			* k
			* (1 + n)
			* np.pi ** (-3 / (2 * n))
			* (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n)
			/ R**2
	) 
	m = M / N  
	pos = np.random.randn(N, 3)  
	vel = np.zeros(pos.shape)

	normal_result = normal.getAcc(pos, vel, m, h, k, n, lmbda, nu)

	return normal_result

def test_accuracy_function(improved_result):
	normal_result = getNormalAccuracy()

	assert len(normal_result) == len(improved_result)
	for i in range(len(normal_result)):
		assert (normal_result[i] == improved_result[i]).all() # check if the values are all the same
