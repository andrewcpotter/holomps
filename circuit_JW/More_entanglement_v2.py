import time
import numpy as np
from circuit_qubit_JW_v2 import Circuit
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.hubbard import FermiHubbardModel, FermiHubbardChain
from scipy.optimize import minimize
from scipy.optimize import dual_annealing, basinhopping
from fermi_hubbard_dmrg import *
import scipy.special as ss
from scipy.integrate import quad

def fermi_hubbard_half_filling_more_entanglement(t_value, U_value, mu_value, ansatz_type, optimizer_choice, global_iter, C_para, params_start_option, params_start_given):
    d = 2
    chimax = 2
    def qub_x(params): return (np.pi/2, 0, params[0])
    def qub_y(params): return (np.pi/2, np.pi/2, params[0])
    def qub_z(params): return (0, 0, params[0])
    def qub_two(params): return (params[0])

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

    def energy(params, circuit, circuit1, Hamiltonian, psi):
        psi = circuit_imps(params, circuit, circuit1)
        E = (Hamiltonian.expectation_value(psi)).real
        return E
	
    def energy_filling_check(params, circuit, circuit1, Hamiltonian, psi):
        psi = circuit_imps(params, circuit, circuit1)
        E = (Hamiltonian.expectation_value(psi)).real
        filling_now = psi.expectation_value("Ntot")
        filling_diff = abs(filling_now[0] + filling_now[1] - 2)
        E_fix_filling = E + filling_diff * C_para
        return E_fix_filling

    rng = np.random.default_rng()
    params = rng.uniform(high=2*np.pi, size=c.n_params + c1.n_params)
    print(len(params))
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
        result = minimize(energy_filling_check, x0=params_start, args=(c, c1, Hamiltonian, psi), method='nelder-mead')
    elif optimizer_choice == 1:
	    result = dual_annealing(energy_filling_check, bounds, args=(c, c1, Hamiltonian, psi), maxiter = global_iter)
    elif optimizer_choice == 2:
        result = basinhopping(energy_filling_check, x0 = params_start, minimizer_kwargs={"method":"nelder-mead","args": (c, c1, Hamiltonian, psi)}, niter = global_iter)
    elif optimizer_choice == 3:
        result = minimize(energy_filling_check, x0=params_start, args=(c, c1, Hamiltonian, psi), method='BFGS')
    elif optimizer_choice == 4:
        result = basinhopping(energy_filling_check, x0 = params_start, minimizer_kwargs={"method":"nelder-mead","args": (c, c1, Hamiltonian, psi)}, niter = global_iter) 
    sweet_spot = result.x
    t2 = time.time()

    print("it took {}s to optimize".format(t2-t1))
    #print("sweet spot = {}".format(sweet_spot))
    print("num function evaluations: {}".format(result['nfev']))
    print("num iterations: {}".format(result['nit']))
    #print("termination msg: {}".format(result['message']))

    def fermi_hubbard_ground_energy(t, U):
        E0_exact = -4 * t * quad(lambda x: (ss.j0(x) * ss.j1(x)) * np.exp(-U*x/2) / (x * (1 + np.exp(-U*x/2))), 0, np.inf)[0]
        return E0_exact
    E0 = fermi_hubbard_ground_energy(t_value, U_value)

    holo_gs = circuit_imps(sweet_spot, c, c1)
    filling_gs = holo_gs.expectation_value("Ntot")
    holo_E = energy(sweet_spot, c, c1, Hamiltonian, holo_gs)
    return (holo_E, filling_gs, sweet_spot)