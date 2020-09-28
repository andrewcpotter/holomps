import time
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
from circuit_type_original import *
from MPS_E_evaluation import *

def fh_VQE(t_value, U_value, mu_value, ansatz_type, optimizer_choice, global_iter, C_para, params_start_option, params_start_given):
    d = 2
    chimax = 2
    if ansatz_type == 0:
        c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c = bond_4_optimize_type(c)
        c1 = bond_4_optimize_type(c1)
    elif ansatz_type == 1:
        c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax), \
        ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax), \
        ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c = star_circuit(c)
        c1 = star_circuit(c1)
    elif ansatz_type == 2:
        c = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax), \
        ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax), \
        ("qubit", "b", chimax), ("qubit", "b", chimax)])
        c = fully_connected_circuit(c)
        c1 = fully_connected_circuit(c1)

    rng = np.random.default_rng()
    params = rng.uniform(high=2*np.pi, size=c.n_params + c1.n_params)
    psi = circuit_imps(params, c, c1)
    print("norm of wave function = {0}".format(psi.norm))
    model_params = dict(L=2, t=t_value, U=U_value, mu=mu_value, bc_MPS="infinite", cons_N = None, cons_Sz=None, verbose=0)
    M = FermiHubbardChain(model_params)
    Hamiltonian = M.calc_H_MPO()
    lw = [0] * len(params)
    up = [2.0*np.pi] * len(params)
    bounds=list(zip(lw,up))
    t1 = time.time()
    if params_start_option == True:
        params_start = params_start_given
    else:
        params_start = params
    if optimizer_choice == 0:
        result = minimize(energy_filling_check, x0=params_start, args=(c, c1, Hamiltonian, psi, C_para), method='nelder-mead')
    elif optimizer_choice == 1:
	    result = dual_annealing(energy_filling_check, bounds, args=(c, c1, Hamiltonian, psi, C_para), maxiter = global_iter)
    elif optimizer_choice == 2:
        result = basinhopping(energy_filling_check, x0 = params_start, \
        minimizer_kwargs={"method":"nelder-mead","args": (c, c1, Hamiltonian, psi, C_para)}, niter = global_iter)
    elif optimizer_choice == 3:
        result = minimize(energy_filling_check, x0=params_start, args=(c, c1, Hamiltonian, psi, C_para), method='BFGS')
    elif optimizer_choice == 4:
        result = basinhopping(energy_filling_check, x0 = params_start, \
        minimizer_kwargs={"method":"nelder-mead","args": (c, c1, Hamiltonian, psi, C_para)}, niter = global_iter) 
    sweet_spot = result.x
    t2 = time.time()
    print("it took {}s to optimize".format(t2-t1))
    print("num function evaluations: {}".format(result['nfev']))
    print("num iterations: {}".format(result['nit']))
    E0 = fermi_hubbard_ground_energy(t_value, U_value)
    holo_gs = circuit_imps(sweet_spot, c, c1)
    filling_gs = holo_gs.expectation_value("Ntot")
    holo_E = energy(sweet_spot, c, c1, Hamiltonian, holo_gs)
    return (holo_E, filling_gs, sweet_spot)