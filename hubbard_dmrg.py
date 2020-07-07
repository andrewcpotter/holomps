import numpy as np

from tenpy.networks.mps import MPS
from circuit_qubit import Circuit
from tenpy.algorithms import dmrg
from tenpy.networks.site import SpinHalfFermionSite, FermionSite
from tenpy.models.fermions_spinless import FermionModel
from mps_stuff_fermion import circuit_imps

def Hubbard_spinless_fermion_dmrg(J_value, mu_value, V_value):
    M = FermionModel({'bc_MPS':'infinite', 'conserve':None, 'J':J_value, 'mu':mu_value, 'V':V_value})
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

