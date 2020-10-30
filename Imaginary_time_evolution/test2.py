# import libraries, to construct the ansatz and calculate derivatives:

# first copy what have been done for TFIM ansatz construction:
# have two qubits, and as less gates as possible, to test how taking derivatives work
import time 
import numpy as np
from circuit_qubit_simplified import Circuit
from mps_stuff import ising_impo, circuit_imps
from scipy.optimize import minimize
from scipy.optimize import dual_annealing

d = 2
chimax = 2
J = 1
g = 1

c = Circuit([("qubit", "p", d), ("qubit", "b", chimax)])
def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

c.add_gate("XX", qids=[0, 1], n_params = 1, fn = qub_two)
c.add_gate("YY", qids=[0, 1], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)
c.assemble()

def energy(params, circuit, Hamiltonian, psi):
    psi = circuit_imps(params, circuit)
    E = (Hamiltonian.expectation_value(psi)).real
    return E

rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi, size=c.n_params)
d_params = 1/1000 * np.ones(c.n_params)
print(d_params)
params_1 = params + d_params
psi = circuit_imps(params, c)
psi1 = circuit_imps(params_1, c)
print(psi)
print(psi1)
print(psi1 - psi)

# Hamiltonian = ising_impo(J, g)
# t1 = time.time()
# result = minimize(energy, x0=params, args=(c, Hamiltonian, psi), method='nelder-mead')
# sweet_spot = result.x
# t2 = time.time()

# print("sweet spot = {}".format(sweet_spot))
# print("num function evaluations: {}".format(result['nfev']))
# print("num iterations: {}".format(result['nit']))
# print("termination msg: {}".format(result['message']))

# from scipy.integrate import quad

# def infinite_gs_energy(J, g):
    # """
    # Straight from tenpy docs: https://tenpy.readthedocs.io/en/latest/intro/examples/tfi_exact.html
    # """
    # def f(k, lambda_):
        # return np.sqrt(1 + lambda_**2 + 2 * lambda_ * np.cos(k))

    # E0_exact = -g / (J * 2. * np.pi) * quad(f, -np.pi, np.pi, args=(J / g, ))[0]
    # return E0_exact

# holo_gs = circuit_imps(sweet_spot, c)
# holo_E = energy(sweet_spot, c, Hamiltonian, holo_gs)
# E0 = infinite_gs_energy(J, g)
# print("holoMPS: E = {:.8}".format(holo_E))
# print(" theory: E = {:.8}".format(E0))
# print("  error: {:.7%}".format(np.abs(1 - holo_E/E0)))