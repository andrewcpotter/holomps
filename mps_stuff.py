#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:52:28 2020

@author: Aaron (tensor shit) , Dhruva (gates stuff)
"""
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
import numpy as np
import scipy.sparse as ss


def circuit_imps(params, circuit, spin=None):
    if spin == None:
        spin = SpinHalfSite(None)
    num_params = circuit.n_params
    unitary = circuit.get_tensor(params)
    B = [np.swapaxes(unitary[:,:,0,:],1,2)]
    psi = MPS.from_Bflat([spin], B, bc='infinite', dtype=complex, form=None)    
    if psi.form is not None:
        try:
            psi.canonical_form()
            psi.convert_form(psi.form)
        except:
            print("psi form thing didn't work")    
    return psi


def circuit_mps(params, circuit, N, bc='finite', spin=None):
    if spin == None:
        spin = SpinHalfSite(None)
    
    # evaluate circuits on params, get rank-4 ([out,out,in,in]) unitary object
    tensors = [circuit.get_tensor(params)]*N
    
    # now change the order of indices to [p, vL, vR] = [p_out,b_in,b_out] from
    # [p_out,b_out,p_in,b_in], (with p_in = 0 to go from unitary to isometry)
    B_arrs = [np.swapaxes(tensor[:,:,0,:],1,2) for tensor in tensors]
    B_arrs[0] = B_arrs[0][:,0:1,:]
    B_arrs[-1] = B_arrs[-1][:,:,0:1]
    
    psi = MPS.from_Bflat([spin]*N, B_arrs, bc, dtype=complex, form=None)    
    if psi.form is not None:
        try:
            psi.canonical_form()
            psi.convert_form(psi.form)
        except:
            print("psi form thing didn't work")    
    return psi

def ising_mpo(J, g, N, bc='finite', spin=None):
    if spin == None:
        spin = SpinHalfSite(None)
    Id, Sp, Sm, Sz = spin.Id, spin.Sp, spin.Sm, 2*spin.Sz
    Sx= Sp + Sm 
    # confirmed from TeNPy docs, lines below directly copied from docs
    W_bulk = [[Id,Sx,g*Sz],[None,None,-J*Sx],[None,None,Id]] 
    W_first = [W_bulk[0]]  # first row
    W_last = [[row[-1]] for row in W_bulk]  # last column
    Ws = [W_first] + [W_bulk]*(N-2) + [W_last]
    H = MPO.from_grids([spin]*N, Ws, bc, IdL=0, IdR=-1) # (probably leave the IdL,IdR)
    return H

def ising_impo(J, g, spin=None):
    if spin == None:
        spin = SpinHalfSite(None)
    Id, Sp, Sm, Sz = spin.Id, spin.Sp, spin.Sm, 2*spin.Sz
    Sx= Sp + Sm 
    # confirmed from TeNPy docs, lines below directly copied from docs
    W_bulk = [[Id,Sx,g*Sz],[None,None,-J*Sx],[None,None,Id]]
    H = MPO.from_grids([spin], [W_bulk], bc='infinite', IdL=0, IdR=-1)
    return H
    
### directly from tenpy docs:
def xxz_mpo(J=1.0, Delta=1.0, hz=0.2, N=4, bc='finite', spin=None):
    if spin == None:
        spin = SpinHalfSite(None)
    Id, Sp, Sm, Sz = spin.Id, spin.Sp, spin.Sm, 2*spin.Sz
    W_bulk = [[Id, Sp, Sm, Sz, -hz * Sz],
              [None, None, None, None, 0.5 * J * Sm],
              [None, None, None, None, 0.5 * J * Sp],
              [None, None, None, None, J * Delta * Sz],
              [None, None, None, None, Id]]
    W_first = [W_bulk[0]]  # first row
    W_last = [[row[-1]] for row in W_bulk]  # last column
    Ws = [W_first] + [W_bulk]*(N - 2) + [W_last]
    H = MPO.from_grids([spin]*N, Ws, bc, IdL=0, IdR=-1) # (probably leave the IdL,IdR)
    return H

### open boundaries
def ising_hamiltonian(J, g, N):
    ## the two functions below are inverses.
    def spinstr(spinint, N):
        return bin(int(spinint))[2:].zfill(N)
    def getint(N, config):
        return int(config.zfill(N), base=2)
    
    try:
        Ham = ss.load_npz("./TFIM_N{}_J{}_g{}.npz".format(N, J, g))
    except:
        vals = []
        rows = []
        cols = []
        Size = 2**N
        for state in range(Size):
            scfg = spinstr(state,N)
            term = 0.0
            for site in range(N-1):
                vals.append(-J)
                rows.append( state^( 2**(N-1-site) + 2**(N-2-site) ) )
                cols.append(state)
                term += g*(2*int(scfg[site])-1)
            term += g*(2*int(scfg[-1])-1) ## final site
            vals.append(term)
            rows.append(state)
            cols.append(state)
        Ham = ss.coo_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                            shape=(Size,Size), dtype=float)
        ss.save_npz("./TFIM_N{}_J{}_g{}.npz".format(N, J, g), Ham)
    return Ham

### note: for XXZ we'll probably want to look in fixed Sz sector? Later for that.
        