import time
import numpy as np
from circuit_qubit_simplified import *
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
import numpy as np
import scipy.sparse as ss

#from tenpy.models.tf_ising import TFIModel
#from qubit_circuit_with_JW1 import *

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

def circuit_imps(params, circuit):
    site = SpinHalfSite(conserve=None)
    # evaluate circuit, get rank-4 (p_out, b_out, p_in, b_in) unitary
    unitary = circuit.get_tensor(params)
    # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # (with p_in = 0 to go from unitary to isometry)
    B = np.swapaxes(unitary[:,:,0,:],1,2)
    psi = MPS.from_Bflat([site], [B], bc='infinite', dtype=complex, form=None)    
    if psi.form is not None:
        try:
            psi.canonical_form()
            psi.convert_form(psi.form)
        except:
            print("psi form thing didn't work")
    return psi

def ising_impo(J, g):
    site = SpinHalfSite(None)
    Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
    Sx = Sp + Sm
    W = [[Id,Sx,g*Sz], [None,None,-J*Sx], [None,None,Id]]
    H = MPO.from_grids([site], [W], bc='infinite', IdL=0, IdR=-1)
    return H

def energy(params, circuit, Hamiltonian):
    psi = circuit_imps(params, circuit)
    E = (Hamiltonian.expectation_value(psi)).real
    return E
	
Hamiltonian = ising_impo(J, g)
	
Entangler_pool = [XXGate(qids = [0,1], fn = qub_two), YYGate(qids = [0,1], fn=qub_two), ZZGate(qids = [0,1], fn=qub_two),\
RotGate(qids=[0], n_params=1, fn=qub_x), RotGate(qids=[1], n_params=1, fn=qub_x),\
RotGate(qids=[0], n_params=1, fn=qub_y), RotGate(qids=[1], n_params=1, fn=qub_y),\
RotGate(qids=[0], n_params=1, fn=qub_z), RotGate(qids=[1], n_params=1, fn=qub_z)]

# the starting point
# set optimize_value to 0
optimize_value = 0
iter_confine1 = 10
iter_confine2 = 10
energy_old = 1000
total_round = 9

gate_list = [XXGate(qids = [0,1], fn = qub_two),YYGate(qids = [0,1], fn = qub_two)]
entangler_choice_p = gate_list[len(gate_list)-1]
#gate_list = [ZZGate(qids = [0,2], fn=qub_two), ZZGate(qids = [1,3], fn=qub_two)]

c = Circuit([("qubit", "p", d),("qubit", "b", chimax)])
for gate in gate_list:
    c.gates.append(gate)
c.assemble()

rng = np.random.default_rng()
params = rng.uniform(high=2*np.pi, size=c.n_params)
print(c.n_params)
print(params)
result = minimize(energy, x0=params, args=(c, Hamiltonian), method='nelder-mead',options={'maxiter':iter_confine2})
sweet_spot = result.x
known_params = sweet_spot 
optimize_value = energy(sweet_spot, c, Hamiltonian)
print(optimize_value)
optimize_value_old = energy_old

def energy_part_vary(params, known_params, c, gate, c_new, Hamiltonian):
    #params_total = known_params[0, c.n_params] + params[0,gate.n_params] + known_params[c.n_params, c.n_params+c1.n_params] +\
    #params[gate.n_params, gate.n_params  + gate1.n_params]
    a1 = known_params[0: c.n_params]
    a2 = params[0: gate.n_params]
    params_total = np.concatenate((a1, a2), axis=0, out=None)
    psi = circuit_imps(params_total, c_new)
    E = (Hamiltonian.expectation_value(psi)).real
    return E

for step_round in range(total_round):
    print("this is round:", step_round)
    optimize_value_old = energy_old
    for gate_choice in Entangler_pool:
        t1 = time.time()
        gate = gate_choice
        gate1 = gate_choice
        params_start = rng.uniform(high=2*np.pi, size = gate.n_params)
        c_new = Circuit([("qubit", "p", d), ("qubit", "b", chimax)])
        gate_list_copy = gate_list.copy()
        gate_list_copy.append(gate_choice)
        for gate in gate_list_copy:
            c_new.gates.append(gate)
        c_new.assemble()
        
        result_new = minimize(energy_part_vary, x0 = params_start, args = (known_params,c,gate,c_new,Hamiltonian), method = 'nelder-mead',\
        options={'maxiter':iter_confine1})
        sweet_spot = result_new.x
        optimize_value_new = energy_part_vary(sweet_spot, known_params, c, gate, c_new, Hamiltonian)
        if optimize_value_new < optimize_value_old and gate_choice != entangler_choice_p:
            optimize_value_old = optimize_value_new
            entangler_choice = gate_choice
            sweet_spot_local = sweet_spot
        t2 = time.time()        
    c_new = Circuit([("qubit", "p", d), ("qubit", "b", chimax)])
    print(entangler_choice)
    print(Entangler_pool.index(entangler_choice))
    gate_list.append(entangler_choice)
	
    entangler_choice_p = entangler_choice
	
    for gate in gate_list:
        c_new.gates.append(gate)
    c_new.assemble()
	
    a1 = known_params[0: c_new.n_params - entangler_choice.n_params]
    a2 = sweet_spot_local[0: entangler_choice.n_params]
	
    params_local_best = np.concatenate((a1, a2), axis=0, out=None)
    params_random = rng.uniform(high=2*np.pi, size = c_new.n_params)
    result = minimize(energy, x0 = params_random, args = (c_new,Hamiltonian), method = 'nelder-mead'\
    ,options={'maxiter':iter_confine2})
    c = c_new
    # result = minimize(energy_filling_check, x0 = params_local_best, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead'\
    # ,options={'maxiter':iter_confine2})
    #result = minimize(energy_filling_check, x0 = params_local_best, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead')
    known_params = result.x
    optimize_value_old = energy(known_params, c_new, Hamiltonian)
    optimize_value_old0 = optimize_value_old + 1
    while abs(optimize_value_old0 - optimize_value_old)>1e-6:
        optimize_value_old0 = optimize_value_old
        result = minimize(energy, x0 = known_params, args = (c_new,Hamiltonian), method = 'nelder-mead',\
        options={'maxiter':iter_confine2*step_round})
        known_params = result.x
        optimize_value_old = energy(known_params, c_new, Hamiltonian)
        print(optimize_value_old)
    t3 = time.time()
    print(optimize_value_old)
    print("it takes", t3 - t1, "seconds to finish this round")

from scipy.integrate import quad

def infinite_gs_energy(J, g):
    """
    Straight from tenpy docs: https://tenpy.readthedocs.io/en/latest/intro/examples/tfi_exact.html
    """
    def f(k, lambda_):
        return np.sqrt(1 + lambda_**2 + 2 * lambda_ * np.cos(k))

    E0_exact = -g / (J * 2. * np.pi) * quad(f, -np.pi, np.pi, args=(J / g, ))[0]
    return E0_exact

E0 = infinite_gs_energy(J, g)
print(" theory: E = {:.8}".format(E0))