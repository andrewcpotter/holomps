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


params_better_list =  [ 6.99725361e+02,  3.92253812e+00,  2.27220347e+00,  3.69775083e+00,
1.39417389e+00,  3.97744791e+00,  4.03549706e+00,  4.31498975e+00,
6.28093169e+00,  2.35802857e+00,  3.28388348e+00,  1.66712653e+00,
1.55927117e+00,  5.68291383e+00,  7.18219555e-03,  2.80636361e-03,
8.59807741e-03, 6.99044578e-02,  5.08178233e-02, 2.49475627e-01,
-3.57429382e-04,  1.02715563e+00,  9.13289877e-03, -1.55603511e-01,
4.32436136e-02, -4.43039682e-02,  1.65162526e-01,  3.63538442e-01,
5.50423121e-02,  1.07553845e-02,  2.80323371e-01, -1.03383905e-02,
-7.01301458e-04,  1.64481520e-01, -7.55169142e-03,  4.08666838e-02,
-2.55816180e-02, -2.01463763e-02,  4.15517251e-01,  1.99215335e-02,
7.66375907e-02,  9.07855950e-02,  5.01762943e-03,  5.49720356e+00,
-1.13097092e+00,  1.77496725e+00,  6.12571871e+00,  2.21968308e+00,
4.39528298e+00,  3.98907634e+00,  2.36756115e+00,  5.42849826e+00,
1.17872187e+00,  3.65357109e+00,  3.59497003e+00,  8.87211811e+00,
2.96510784e-01,  4.75052039e-02,  9.71672632e-03, -1.31763224e-01,
-1.86411339e-01, -1.14984681e-01, -2.52331735e-03,  1.55799358e-01,
-2.91032157e+01,  2.94974922e-01, -1.74253463e-01,  1.44999638e-04,
-1.41225495e-03, -1.10184595e-02,  4.95489506e-01, -1.15710688e-05,
3.31051721e-01,  5.11136598e-03, -2.60112089e-02,  1.30042174e-03,
-5.69770173e-03,  2.94701372e-01,  2.89926075e+01,  3.40660903e-05,
-3.78675058e-02,  1.73814139e-03, -5.51185379e-03,  1.51566870e-02]

params_better = np.array(params_better_list)

circuit_list = [("qubit", "p", d)] * 2 + [("qubit", "b", chimax)] * 4
c = Circuit(circuit_list)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)
	
c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[2, 3], n_params = 1, fn = qub_two)
	
c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)
	
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)
	
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = qub_two)
	
c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[4, 5], n_params = 1, fn = qub_two)
	
c.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = qub_two)

c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)
	
c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_z)

c.add_gate("XX_YY", qids=[2, 4], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[2, 4], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[3, 5], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[3, 5], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[2, 3], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[4, 5], n_params = 1, fn = qub_two)

c.add_gate("XX_YY", qids=[2, 4], n_params = 1, fn = qub_two)
c.add_gate("ZZ", qids=[2, 4], n_params = 1, fn = qub_two)
c.add_gate("XX_YY", qids=[3, 5], n_params = 1, fn = qub_two)    
c.add_gate("ZZ", qids=[3, 5], n_params = 1, fn = qub_two)

c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_z)

c.assemble()

circuit_list = [("qubit", "p", d)] * 2 + [("qubit", "b", chimax)] * 4
c1 = Circuit(circuit_list)
c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c1.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[2, 3], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = qub_two)

c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)
	
c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c1.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[4, 5], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = qub_two)

c1.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c1.add_gate("rotation", qids = [2], n_params = 1, fn = qub_z)

c1.add_gate("XX_YY", qids=[2, 4], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[2, 4], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[3, 5], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[3, 5], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[2, 3], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[4, 5], n_params = 1, fn = qub_two)

c1.add_gate("XX_YY", qids=[2, 4], n_params = 1, fn = qub_two)
c1.add_gate("ZZ", qids=[2, 4], n_params = 1, fn = qub_two)
c1.add_gate("XX_YY", qids=[3, 5], n_params = 1, fn = qub_two)    
c1.add_gate("ZZ", qids=[3, 5], n_params = 1, fn = qub_two)

c1.add_gate("rotation", qids = [2], n_params = 1, fn = qub_z)

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
total_site = 25

psi_optimize = circuit_imps(params_better, c, c1)

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