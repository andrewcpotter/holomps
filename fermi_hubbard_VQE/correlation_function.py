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
import matplotlib.pyplot as plt

d = 2
chimax = 2
ansatz_type = 2
def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

params_better_list = [-256.23463238,    5.5112596,     2.16164228,    0.44755776,    1.92864367,
1.17175839,    0.41028707,    5.10834304,    4.65043006,    3.92278362,
1.16931763,    2.86851214,    2.94505863,    3.21385291,   -4.75280756,
5.70473526,    4.75276501,    0.93719745,    4.54496849,    0.64285625,
2.61987153,    1.96641333,    4.07507241,    5.17909908,    0.29662942,
0.69579681,    3.41605951,    3.24926783]
params_better = np.array(params_better_list)

c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
    # one qubit rotation
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)

if ansatz_type == 0:
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = qub_two)
elif ansatz_type == 1:
    c.add_gate("XX_YY", qids=[0, 3], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [0, 3], n_params = 1, fn = qub_two)
    c.add_gate("XX_YY", qids=[1, 2], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [1, 2], n_params = 1, fn = qub_two)
elif ansatz_type == 2:
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = qub_two)
    c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = qub_two)
	
c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)
# one qubit rotation
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)
c.assemble()


# two-site part:
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
# one qubit rotation
c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c1.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)
	
if ansatz_type == 0:
    c1.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)
    c1.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = qub_two)
elif ansatz_type == 1:
    c1.add_gate("XX_YY", qids=[0, 3], n_params = 1, fn = qub_two)
    c1.add_gate("ZZ", qids = [0, 3], n_params = 1, fn = qub_two)
    c1.add_gate("XX_YY", qids=[1, 2], n_params = 1, fn = qub_two)
    c1.add_gate("ZZ", qids = [1, 2], n_params = 1, fn = qub_two)
elif ansatz_type == 2:
    c1.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)
    c1.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = qub_two)
    c1.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)
    c1.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = qub_two)
		
c1.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)

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
	
start_position = 100
total_site = 20

psi_optimize = circuit_imps(params_better, c, c1)

list(range(start_position, start_position + total_site + 1))

psi_SzSz_list0 = psi_optimize.correlation_function("Sz", "Sz", sites1=[start_position], \
sites2=list(range(start_position+1, start_position + total_site + 1)),\
opstr=None, str_on_first=True, hermitian=False, autoJW=True)

psi_NtotNtot_list0 = psi_optimize.correlation_function("Ntot", "Ntot", sites1=[start_position], \
sites2=list(range(start_position+1, start_position + total_site + 1)),\
opstr=None, str_on_first=True, hermitian=False, autoJW=True)

psi_SzSz_list = psi_SzSz_list0[0]
psi_NtotNtot_list = psi_NtotNtot_list0[0]


psi_Sz_list = psi_optimize.expectation_value('Sz', sites = \
list(range(start_position, start_position + total_site + 1)))

psi_Ntot_list = psi_optimize.expectation_value('Ntot', sites = \
list(range(start_position, start_position + total_site + 1)))


SzSz_array = np.zeros(total_site)
NtotNtot_array = np.zeros(total_site)

r_list = list(range(1, total_site+1))
for m in range(total_site):
    SzSz_array[m] = (psi_SzSz_list[m] - psi_Sz_list[0] * psi_Sz_list[m+1])
    NtotNtot_array[m] = (psi_NtotNtot_list[m] - psi_Ntot_list[0] * psi_Ntot_list[m+1])
	
SzSz_list = SzSz_array.tolist()
NtotNtot_list = NtotNtot_array.tolist()

print(r_list)
print(SzSz_list)
print(NtotNtot_list)