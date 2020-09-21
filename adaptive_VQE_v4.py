import time
import numpy as np
from circuit_qubit import *
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.hubbard import FermiHubbardModel, FermiHubbardChain
from scipy.optimize import minimize
from hubbard_dmrg import *

d = 2
chimax = 2
t_value = 1.0
mu_value = 0
U_value = 8.0

model_params = dict(L=2, t=t_value, U=U_value, mu=mu_value, bc_MPS="infinite", cons_N = None, cons_Sz=None, verbose=0)
M = FermiHubbardChain(model_params)
Hamiltonian = M.calc_H_MPO()
rng = np.random.default_rng()

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

def energy(params, circuit, circuit1, Hamiltonian):
    psi = circuit_imps(params, circuit, circuit1)
    E = (Hamiltonian.expectation_value(psi)).real
    return E

def energy_filling_check(params, circuit, circuit1, Hamiltonian):
    psi = circuit_imps(params, circuit, circuit1)
    E = (Hamiltonian.expectation_value(psi)).real
    filling_now = psi.expectation_value("Ntot")
    filling_diff = abs(filling_now[0] + filling_now[1] - 2)
    E_fix_filling = E + filling_diff * 10
    return E_fix_filling

# if bond dim = 4, the total number of gates in the entangler pole is 16;
# 16 = C^2_4 (for XX_YY and ZZ gates) * 2 + 4
# with bond dim increasing to 16, the number of gates in the entangler pole increases drastically
def qub_z(params): return (0,0,params[0])
def qub_two(params): return (params[0])

Entangler_pool = [XX_YYGate(qids = [0,1], fn = qub_two),XX_YYGate(qids = [0,2], fn=qub_two),XX_YYGate(qids = [0,3], fn=qub_two),\
XX_YYGate(qids = [1,2], fn = qub_two),XX_YYGate(qids = [1,3], fn = qub_two),XX_YYGate(qids = [2,3], fn = qub_two),\
ZZGate(qids = [0,1], fn = qub_two), ZZGate(qids = [0,2], fn = qub_two), ZZGate(qids = [0,3], fn = qub_two),\
ZZGate(qids = [1,2], fn = qub_two), ZZGate(qids = [1,3], fn = qub_two), ZZGate(qids = [2,3], fn = qub_two),\
RotGate(qids=[0], n_params=1, fn=qub_z), RotGate(qids=[1], n_params=1, fn=qub_z),\
RotGate(qids=[2], n_params=1, fn=qub_z), RotGate(qids=[3], n_params=1, fn=qub_z)]

# the starting point
c0 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])

# set optimize_value to 0
optimize_value = 0
iter_confine1 = 4
iter_confine2 = 1000
energy_old = 1000
total_round = 20

entangler_choice = XX_YYGate(qids = [0,2], fn = qub_two)

gate_list = [XX_YYGate(qids = [0,2], fn=qub_two), ZZGate(qids = [0,2], fn = qub_two),\
XX_YYGate(qids = [1,3], fn=qub_two), ZZGate(qids = [1,3], fn = qub_two)]

c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
for gate in gate_list:
    c.gates.append(gate)
    c1.gates.append(gate)
c.assemble()
c1.assemble()

rng = np.random.default_rng()
params = rng.uniform(high=2*np.pi, size=c.n_params + c1.n_params)
psi = circuit_imps(params, c, c1)
result = minimize(energy_filling_check, x0=params, args=(c, c1, Hamiltonian), method='nelder-mead',options={'maxiter':iter_confine2})
sweet_spot = result.x
known_params = sweet_spot 
optimize_value = energy(sweet_spot, c, c1, Hamiltonian)
print(optimize_value)
optimize_value_old = energy_old

def energy_part_vary(params, known_params, c, c1, gate, gate1, c_new, c1_new, Hamiltonian):
    #params_total = known_params[0, c.n_params] + params[0,gate.n_params] + known_params[c.n_params, c.n_params+c1.n_params] +\
    #params[gate.n_params, gate.n_params  + gate1.n_params]
    a1 = known_params[0: c.n_params]
    a2 = params[0: gate.n_params]
    a3 = known_params[c.n_params: c.n_params+c1.n_params]
    a4 = params[gate.n_params: gate.n_params+gate1.n_params]
    params_total = np.concatenate((a1, a2, a3, a4), axis=0, out=None)
    psi = circuit_imps(params_total, c_new, c1_new)
    E = (Hamiltonian.expectation_value(psi)).real
    filling_now = psi.expectation_value("Ntot")
    filling_diff = abs(filling_now[0] + filling_now[1] - 2)
    E_fix_filling = E + filling_diff * 10
    return E_fix_filling

for step_round in range(total_round):
    print("this is round:", step_round)
    optimize_value_old = energy_old
    for gate_choice in Entangler_pool:
        t1 = time.time()
        gate = gate_choice
        gate1 = gate_choice
        params_start = rng.uniform(high=2*np.pi, size = gate.n_params + gate1.n_params)
        c_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c1_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
        gate_list_copy = gate_list.copy()
        gate_list_copy.append(gate_choice)
        for gate in gate_list_copy:
            c_new.gates.append(gate)
            c1_new.gates.append(gate)
        c_new.assemble()
        c1_new.assemble()
        
        result_new = minimize(energy_part_vary, x0 = params_start, args = (known_params,c,c1,gate,gate1,c_new, c1_new,Hamiltonian), method = 'nelder-mead',\
        options={'maxiter':iter_confine1})
        sweet_spot = result_new.x
        optimize_value_new = energy_part_vary(sweet_spot, known_params, c, c1, gate, gate1, c_new, c1_new, Hamiltonian)
        if optimize_value_new < optimize_value_old:
            optimize_value_old = optimize_value_new
            entangler_choice = gate_choice
            sweet_spot_local = sweet_spot
        t2 = time.time()        
    c_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
    c1_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
    gate_list.append(entangler_choice)
    for gate in gate_list:
        c_new.gates.append(gate)
        c1_new.gates.append(gate)
    c_new.assemble()
    c1_new.assemble()
	
    a1 = known_params[0: c_new.n_params - entangler_choice.n_params]
    a2 = sweet_spot_local[0: entangler_choice.n_params]
    a3 = known_params[c_new.n_params - entangler_choice.n_params: c1_new.n_params + c_new.n_params -2*entangler_choice.n_params]
    a4 = sweet_spot_local[entangler_choice.n_params: 2 * entangler_choice.n_params]
	
    params_local_best = np.concatenate((a1, a2, a3, a4), axis=0, out=None)
    params_random = rng.uniform(high=2*np.pi, size = c_new.n_params + c1_new.n_params)
    result = minimize(energy_filling_check, x0 = params_random, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead'\
    ,options={'maxiter':iter_confine2})
    c = c_new
    c1 = c1_new
    known_params = result.x
    optimize_value = energy(known_params, c_new, c1_new, Hamiltonian)
    if optimize_value_old < optimize_value:
        result = minimize(energy_filling_check, x0 = params_local_best, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead')
        known_params = result.x
        optimize_value_old = energy(known_params, c_new, c1_new, Hamiltonian)
    else:
        optimize_value_old = optimize_value
	
    t3 = time.time()
    print(optimize_value_old)
    print("it takes", t3 - t1, "seconds to finish this round")