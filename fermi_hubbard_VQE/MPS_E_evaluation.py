import numpy as np
from circuit_qubit import Circuit
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.hubbard import FermiHubbardModel, FermiHubbardChain
from scipy.optimize import minimize
from scipy.optimize import dual_annealing, basinhopping
from fermi_hubbard_dmrg import *
import scipy.special as ss
from scipy.integrate import quad

# def circuit_imps(params, circuit, circuit1):
    # site = SpinHalfFermionSite(cons_N = None, cons_Sz = None, filling = 1)
    # # evaluate circuit, get rank-4 (p_out, b_out, p_in, b_in) unitary
    # params0 = params[0:circuit.n_params]
    # params1 = params[circuit.n_params: circuit.n_params + circuit1.n_params]
    # unitary0 = circuit.get_tensor(params0)
    # unitary1 = circuit1.get_tensor(params1)
    # # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # # (with p_in = 0 to go from unitary to isometry)
    # B0 = [np.swapaxes(unitary0[:,:,1,:],1,2)]
    # B1 = [np.swapaxes(unitary1[:,:,2,:],1,2)]
    # psi = MPS.from_Bflat([site]*2, B0+B1, bc="infinite", dtype=complex, form=None)
    # if psi.form is not None:
        # try:
            # psi.canonical_form()
            # psi.convert_form(psi.form)
        # except:
            # print("psi form thing didn't work")
    # return psi
    
def circuit_imps(params, circuit, circuit1):
    site = SpinHalfFermionSite(cons_N = None, cons_Sz = None, filling = 1)
    # evaluate circuit, get rank-4 (p_out, b_out, p_in, b_in) unitary
    params0 = params[0:circuit.n_params]
    params1 = params[circuit.n_params: circuit.n_params + circuit1.n_params]
    trans_matrix = np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]])
    trans_inv = np.linalg.inv(trans_matrix)
    trans_whole = np.multiply.outer(trans_matrix, trans_matrix)
    trans_whole_inv = np.multiply.outer(trans_inv, trans_inv)
    unitary0 = circuit.get_tensor(params0)
    unitary1 = circuit1.get_tensor(params1)
    unitary0new = np.matmul(np.matmul(trans_whole_inv, unitary0), trans_whole)
    unitary1new = np.matmul(np.matmul(trans_whole_inv, unitary1), trans_whole)
    # print(unitary0new)
    # print(unitary1new)
    # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # (with p_in = 0 to go from unitary to isometry)
    B0 = [np.swapaxes(unitary0new[:,:,1,:],1,2)]
    B1 = [np.swapaxes(unitary1new[:,:,2,:],1,2)]
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
	
def energy_filling_check(params, circuit, circuit1, Hamiltonian, psi, C_para):
    psi = circuit_imps(params, circuit, circuit1)
    E = (Hamiltonian.expectation_value(psi)).real
    filling_now = psi.expectation_value("Ntot")
    filling_diff = abs(filling_now[0] + filling_now[1] - 2)
    E_fix_filling = E + filling_diff * C_para
    return E_fix_filling
    
def fermi_hubbard_ground_energy(t, U):
    E0_exact = -4 * t * quad(lambda x: (ss.j0(x) * ss.j1(x)) * np.exp(-U*x/2) / (x * (1 + np.exp(-U*x/2))), 0, np.inf)[0]
    return E0_exact