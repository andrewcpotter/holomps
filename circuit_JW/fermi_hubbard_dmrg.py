import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.networks.site import SpinHalfFermionSite, FermionSite
from tenpy.models.fermions_spinless import FermionModel
from tenpy.models.hubbard import FermiHubbardModel, FermiHubbardChain

def Hubbard_spinless_fermion_dmrg(t_value, mu_value, V_value):
    M = FermionModel({'bc_MPS':'infinite', 'conserve':None, 'J':t_value, 'mu':mu_value, 'V':V_value})
    product_state = ["empty", "full"]
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc = M.lat.bc_MPS)

    dmrg_params = {
        'mixer': True,
	    'trunc_params':{
	        'chi_max':30,
		    'svd_min':1.e-10,
	    },
	    'max_E_err':1e-10,
	    'verbose':True,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    return E

def Hubbard_spinhalf_fermion_dmrg(t_value, U_value, mu_value, chi_value):
    model_params = dict(L=2, t=t_value, U=U_value, mu=mu_value, bc_MPS = "infinite", cons_N = 'N', cons_Sz=None, verbose=True)
    M = FermiHubbardChain(model_params)
    site = SpinHalfFermionSite(cons_N = 'N', cons_Sz = None, filling = 1)
    product_state = ["up", "down"] * (M.lat.N_sites // 2)
    psi = MPS.from_product_state([site]*M.lat.N_sites, product_state, bc = M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,
        'diag_method': 'ED_block',
        #'min_sweeps': 200,
	    'trunc_params':{
            'chi_max':chi_value,
            'svd_min':1.e-12,
	    },
        'lanczos_params':{
            'P_tol':1e-14,
            'E_tol_to_trunc':1e-12,
            'P_tol_to_truc':1e-14
        },
        'max_E_err':1e-12,
        'verbose':True,
    }
    #eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
    #eng = dmrg.DMRGEngine(psi, M, dmrg_params)
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    filling_gs = psi.expectation_value("Ntot")
    print(filling_gs)
    return (E, filling_gs)
	
def Hubbard_spinhalf_fermion_dmrg_run(t_value, U_value, mu_value, chi_value):
    model_params = dict(L=2, t=t_value, U=U_value, mu=mu_value, bc_MPS = "infinite", cons_N = 'N', cons_Sz=None, verbose=True)
    M = FermiHubbardChain(model_params)
    site = SpinHalfFermionSite(cons_N = 'N', cons_Sz = None, filling = 1)
    product_state = ["up", "up"] * (M.lat.N_sites // 2)
    psi = MPS.from_product_state([site]*M.lat.N_sites, product_state, bc = M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,
	    'trunc_params':{
	        'chi_max':chi_value,
		    'svd_min':1.e-12,
            'P_tol_to_truc':1e-10
	    },
	    'max_E_err':1e-12,
	    'verbose':True,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    filling_gs = psi.expectation_value("Ntot")
    print(filling_gs)
    return E