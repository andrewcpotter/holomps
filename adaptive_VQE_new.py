import time
import numpy as np
from circuit_qubit import Circuit
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.hubbard import FermiHubbardModel, FermiHubbardChain
from scipy.optimize import minimize
from scipy.optimize import dual_annealing, basinhopping
from hubbard_dmrg import *
import scipy.special as ss
from scipy.integrate import quad

d = 2
chimax = 2
t_value = 1.0
mu_value = 0
U_value = 8.0

def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
# one qubit rotation
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[0, 3], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 3], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[1, 2], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 2], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)

#c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)
#c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = qub_two)

# one qubit rotation
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)
c.assemble()


# two-site part:
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
# one qubit rotation
c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c1.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[0, 3], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 3], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[1, 2], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[1, 2], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)

#c1.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)
#c1.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = qub_two)

# one qubit rotation
c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)
c1.assemble()

def circuit_imps(params, circuit, circuit1):
    site = SpinHalfFermionSite(cons_N = None, cons_Sz = None, filling = 1)
    # evaluate circuit, get rank-4 (p_out, b_out, p_in, b_in) unitary
    params0 = params[0:circuit.n_params]
    params1 = params[circuit.n_params: circuit.n_params + circuit1.n_params]
    unitary0 = circuit.get_tensor(params0)
    unitary1 = circuit1.get_tensor(params1)
    # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # (with p_in = 0 to go from unitary to isometry)
    B0 = [np.swapaxes(unitary0[:,:,1,:],1,2)]
    B1 = [np.swapaxes(unitary1[:,:,2,:],1,2)]
    psi = MPS.from_Bflat([site]*2, B0+B1, bc="infinite", dtype=complex, form=None)
    if psi.form is not None:
        try:
            psi.canonical_form()
            psi.convert_form(psi.form)
        except:
            print("psi form thing didn't work")
    return psi

def energy(params, circuit, circuit1, Hamiltonian, psi):
    psi = circuit_imps(params, circuit, circuit1)
    E = (Hamiltonian.expectation_value(psi)).real
    return E
	
def energy_filling_check(params, circuit, circuit1, Hamiltonian, psi):
    psi = circuit_imps(params, circuit, circuit1)
    E = (Hamiltonian.expectation_value(psi)).real
    filling_now = psi.expectation_value("Ntot")
    #filling_diff = abs(filling_now[0] - 1) + abs(filling_now[1] - 1)
    filling_diff = abs(filling_now[0] + filling_now[1] - 2)
    E_fix_filling = E + filling_diff * 10
    return E_fix_filling

def filling_check(params, circuit, circuit1, Hamiltonian, psi):
    psi = circuit_imps(params, circuit, circuit1)
    filling_now = psi.expectation_value("Ntot")
    #filling_diff = abs(filling_now[0] - 1) + abs(filling_now[1] - 1)
    filling_diff = abs(filling_now[0] - 1) + abs(filling_now[1] - 1)
    return filling_diff

rng = np.random.default_rng()
params = rng.uniform(high=2*np.pi, size=c.n_params + c1.n_params)
psi = circuit_imps(params, c, c1)
print("norm of wave function = {0}".format(psi.norm))
print()
model_params = dict(L=2, t=t_value, U=U_value, mu=mu_value, bc_MPS="infinite", cons_N = None, cons_Sz=None, verbose=True)
M = FermiHubbardChain(model_params)
Hamiltonian = M.calc_H_MPO()
lw = [0] * len(params)
up = [2.0*np.pi] * len(params)
bounds=list(zip(lw,up))
t1 = time.time()
#result = minimize(energy_filling_check, x0=params, args=(c, c1, Hamiltonian, psi), method='nelder-mead')
#params_value = result.x
#result = minimize(energy, x0=params, args=(c, c1, Hamiltonian, psi), method='nelder-mead')
#sweet_spot0 = result.x
#result = minimize(filling_check, x0=sweet_spot0, args=(c, c1, Hamiltonian, psi), method='nelder-mead')
result = minimize(energy_filling_check, x0=params, args=(c, c1, Hamiltonian, psi), method='nelder-mead', options={'maxiter':100})
#result = minimize(energy_filling_check, x0=sweet_spot0, args=(c, c1, Hamiltonian, psi), method='nelder-mead')
#result = dual_annealing(energy_filling_check, bounds, args=(c, c1, Hamiltonian, psi), maxiter = 1000)
#result = basinhopping(energy_filling_check, x0 = params, minimizer_kwargs={"method":"nelder-mead","args": (c, c1, Hamiltonian, psi)}, niter = 20)
sweet_spot = result.x
t2 = time.time()

print("it took {}s to optimize".format(t2-t1))
print("sweet spot = {}".format(sweet_spot))
print("num function evaluations: {}".format(result['nfev']))
print("num iterations: {}".format(result['nit']))
print("termination msg: {}".format(result['message']))

def fermi_hubbard_ground_energy(t, U):
    E0_exact = -4 * t * quad(lambda x: (ss.j0(x) * ss.j1(x)) * np.exp(-U*x/2) / (x * (1 + np.exp(-U*x/2))), 0, np.inf)[0]
    return E0_exact
E0 = fermi_hubbard_ground_energy(t_value, U_value)

holo_gs = circuit_imps(sweet_spot, c, c1)
filling_gs = holo_gs.expectation_value("Ntot")
print(filling_gs)
holo_E = energy(sweet_spot, c, c1, Hamiltonian, holo_gs)
print("holoMPS: E = {:.8}".format(holo_E))
print(" theory: E = {:.8}".format(E0))
print("  error: {:.7%}".format(np.abs(1 - holo_E/E0)))

#[5.93799862 1.48395961 2.41664817 4.71224714 3.26529962 3.14150461
# 4.72976106 0.61843255 1.17199492 0.7511318  3.8658839  3.14157692
# 2.79014761 3.14159375 2.63867776 4.84526952]