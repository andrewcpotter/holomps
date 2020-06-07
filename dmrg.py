import numpy as np
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.tf_ising import TFIChain
from tenpy.linalg import random_matrix as randmat
from tenpy.models.spins import SpinModel

# DMRG for TFIM
def dmrg_TFIM(L_val, J_val, g_val, chi_max, psi):
    model_params = dict(L=L_val, J=J_val, g=g_val, bc_MPS='finite', conserve=None, verbose=True)
    M = TFIChain(model_params)
    spin = SpinHalfSite(None) # or: 'parity' or 'Sz' instead of None
    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10
        },
        'verbose': True,
        'combine': True
    }
    psi = rand_mps()
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    return E

# DMRG for XXZ spin chain (finite)
def dmrg_XXZ(L_val, Jx_val, Jy_val, Jz_val, chi_max, psi):
    model_params = dict(
        L=L_val,
        S=0.5,
        Jx=Jx_val,
        Jy=Jy_val,
        Jz=Jz_val,
        bc_MPS='finite',
        conserve=None,
        verbose=True)
    dmrg_params = {
        'mixer': True,
        'trunc_params':{
            'chi_max': chi_max,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
        'verbose': True,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    return E
