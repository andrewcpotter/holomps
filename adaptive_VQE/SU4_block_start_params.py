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

SU4_01 = [XX_YYGate(qids=[0,1],fn=qub_two),ZZGate(qids=[0,1],fn=qub_two)]
SU4_02 = [XX_YYGate(qids=[0,2],fn=qub_two),ZZGate(qids=[0,2],fn=qub_two)]
SU4_03 = [XX_YYGate(qids=[0,3],fn=qub_two),ZZGate(qids=[0,3],fn=qub_two)]
SU4_12 = [XX_YYGate(qids=[1,2],fn=qub_two),ZZGate(qids=[1,2],fn=qub_two)]
SU4_13 = [XX_YYGate(qids=[1,3],fn=qub_two),ZZGate(qids=[1,3],fn=qub_two)]
SU4_23 = [XX_YYGate(qids=[2,3],fn=qub_two),ZZGate(qids=[2,3],fn=qub_two)]
RotSet = [RotGate(qids=[0], n_params=1, fn=qub_z)]

Entangler_pool = [SU4_01, SU4_02, SU4_03, SU4_12, SU4_13, SU4_23, RotSet]

# the starting point
c0 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])

# set optimize_value to 0
optimize_value = 0
iter_confine1 = 4
iter_confine2 = 20
energy_old = 1000
total_round = 20

entangler_choice = XX_YYGate(qids = [0,2], fn = qub_two)
gate_list = [RotGate(qids = [0], n_params=1, fn=qub_z), \
XX_YYGate(qids=[0,2],fn=qub_two), ZZGate(qids=[0,2],fn=qub_two),\
XX_YYGate(qids=[1,3],fn=qub_two), ZZGate(qids=[1,3],fn=qub_two),\
XX_YYGate(qids=[0,1],fn=qub_two), ZZGate(qids=[0,1],fn=qub_two),\
XX_YYGate(qids=[2,3],fn=qub_two), ZZGate(qids=[2,3],fn=qub_two),\
XX_YYGate(qids=[0,2],fn=qub_two), ZZGate(qids=[0,2],fn=qub_two),\
XX_YYGate(qids=[1,3],fn=qub_two), ZZGate(qids=[1,3],fn=qub_two),\
RotGate(qids = [0], n_params=1, fn=qub_z)]
entangler_choice_p = gate_list[len(gate_list)-1]
print(entangler_choice_p)

c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
for gate in gate_list:
    c.gates.append(gate)
    c1.gates.append(gate)
c.assemble()
c1.assemble()

known_params_list = [-0.00884485,  3.92951401,  1.8386671,   3.7941439,   1.41748672,  3.94470862,
4.06581259,  4.34431101,  6.27061076,  2.35540115,  3.512422,  1.75812945,
1.80626837,  5.41031803,  0.01143065,  5.49768665, -1.20902689,  1.58978879,
6.42173957,  2.26759073,  4.28232851,  4.04270491,  2.48656851,  5.49752545,
1.08900805,  3.87478665,  4.2956721,   8.5598388]

known_params = np.array(known_params_list)
optimize_value = -0.27417629275507616
print(optimize_value)
optimize_value_old = energy_old

def energy_part_vary(params, known_params, c, c1, gate_num, c_new, c1_new, Hamiltonian):
    #params_total = known_params[0, c.n_params] + params[0,gate.n_params] + known_params[c.n_params, c.n_params+c1.n_params] +\
    #params[gate.n_params, gate.n_params  + gate1.n_params]
    a1 = known_params[0: c.n_params]
    a2 = params[0: gate_num]
    a3 = known_params[c.n_params: c.n_params+c1.n_params]
    a4 = params[gate_num: 2*gate_num]
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
    for gate_set in Entangler_pool:
        t1 = time.time()
        gate_num = len(gate_set)
        params_start = rng.uniform(high=2*np.pi, size = 2*gate_num)
        c_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c1_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
        gate_list_copy = gate_list.copy()
        for gate in gate_set:
            gate_list_copy.append(gate)
        for gate in gate_list_copy:
            c_new.gates.append(gate)
            c1_new.gates.append(gate)
        c_new.assemble()
        c1_new.assemble()
        
        result_new = minimize(energy_part_vary, x0 = params_start, \
        args = (known_params,c,c1,gate_num,c_new, c1_new,Hamiltonian), \
        method = 'nelder-mead',options={'maxiter':iter_confine1})
        sweet_spot = result_new.x
        optimize_value_new = energy_part_vary(sweet_spot, known_params, c, c1, gate_num, c_new, c1_new, Hamiltonian)
        if optimize_value_new < optimize_value_old and gate_set[len(gate_set)-1] != entangler_choice_p:
            optimize_value_old = optimize_value_new
            entangler_choice = gate_set
            sweet_spot_local = sweet_spot
        t2 = time.time()        
    c_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
    c1_new = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
    print(entangler_choice)
    print(Entangler_pool.index(entangler_choice))
    
    gate_num = len(entangler_choice)

    for gate in entangler_choice:
        gate_list.append(gate)
    entangler_choice_p = entangler_choice[len(entangler_choice)-1]

    for gate in gate_list:
        c_new.gates.append(gate)
        c1_new.gates.append(gate)
    c_new.assemble()
    c1_new.assemble()
	
    a1 = known_params[0: c_new.n_params - gate_num]
    a2 = sweet_spot_local[0: gate_num]
    a3 = known_params[c_new.n_params - gate_num: c1_new.n_params + c_new.n_params -2*gate_num]
    a4 = sweet_spot_local[gate_num: 2 * gate_num]
	
    params_local_best = np.concatenate((a1, a2, a3, a4), axis=0, out=None)
    params_random = rng.uniform(high=2*np.pi, size = c_new.n_params + c1_new.n_params)
    result = minimize(energy_filling_check, x0 = params_random, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead'\
    ,options={'maxiter':iter_confine2})
    c = c_new
    c1 = c1_new
    # result = minimize(energy_filling_check, x0 = params_local_best, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead'\
    # ,options={'maxiter':iter_confine2})
    #result = minimize(energy_filling_check, x0 = params_local_best, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead')
    known_params = result.x
    optimize_value_old = energy(known_params, c_new, c1_new, Hamiltonian)
    print(optimize_value_old)
    optimize_value_old0 = optimize_value_old + 1
    while abs(optimize_value_old0 - optimize_value_old) > 1e-4:
        optimize_value_old0 = optimize_value_old
        result = minimize(energy_filling_check, x0 = known_params, args = (c_new,c1_new,Hamiltonian), method = 'nelder-mead',\
        options={'maxiter':iter_confine2*step_round})
        known_params = result.x
        optimize_value_old = energy(known_params, c_new, c1_new, Hamiltonian)
        print(optimize_value_old)
    t3 = time.time()
    print("it takes", t3 - t1, "seconds to finish this round")