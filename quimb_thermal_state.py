

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  05 03:40:32 2021
@author: shahin75
"""

def prob_list(circuit,params,T):
    """ 
    Returns list of probability weights.
    """
    d = circuit[0,:,0,0].size
    prob_params = params[:d]
    if T != 0:
        exc_list= np.array([np.exp(-k/T) for k in prob_params])
        z = sum(exc_list)
        prob_list = exc_list/z         
    else:
        prob_list = np.zeros(d)
        # setting probability of |0> to 1 at T = 0
        prob_list[0] += 1
    return prob_list

def MPDO(circuit,params,Hamiltonian,T,n,j):
    """
    Constructs the Matrix Product Density Operator
    """
    inds1 = f'p{j}', f'b{n+1}',f'p{j+1}',f'b{n}'
    k = qtn.Tensor(circuit, inds1, tags='k')
    
    data = np.diag(prob_list(circuit,params,T))
    inds2 = f'p{j+1}',f'p{j+2}'
    p_chain = qtn.Tensor(data,inds2,tags='p')

    inds3 = f'p{j+3}',f'bc{n+1}',f'p{j+2}',f'bc{n}'
    b = qtn.Tensor(circuit.conj(), inds3, tags= 'kc')

    inds4 = f'p{j}',f'H{n+1}',f'p{j+3}',f'H{n}'
    H = qtn.Tensor(Hamiltonian,inds4,tags='H')
    
    TN = k & p_chain & b & H 
    return TN

def contractions(unit_list,params,
                 Hamiltonian,L,
                 state_vcl,state_vcr,
                 H_bvecl,H_bvecr,T):
    """
    Returns tensor contractions
    """
    unit_list1 = L * unit_list
    params1 = L * params
    N = L * len(unit_list)
    TN_list = [MPDO(unit_list1[n],params1[n],Hamiltonian,T,n,j) for n,j in zip(range(N),range(0,4*N+6,5))]

    # boundary conditions
    # for ket:
    inds0 = 'b0',
    s_left = qtn.Tensor(state_vcl,inds0,tags='sl')
    inds1 = f'b{N}',f'bc{N}'
    s_right = qtn.Tensor(np.eye(state_vcl.size),inds1,tags='rvec')

    # for bra:
    inds2 = 'bc0',
    sc_left = qtn.Tensor(state_vcl.conj(),inds2,tags='scl')

    # for Hamiltonian
    inds4 = 'H0',
    H_left = qtn.Tensor(H_bvecl,inds4,tags='Hl')  
    inds5 = f'H{N}',
    H_right = qtn.Tensor(H_bvecr,inds5,tags='Hr') 

    # tenor contractions
    TN0 = TN_list[0]
    for j in range(1,len(TN_list)):
        TN0 = TN_list[j] & TN0
    
    TN = s_left & s_right & sc_left & H_left & H_right & TN0
    return TN

def entropy(prob_list):
    """
    Returns the entropy
    """ 
    # avoiding NaN in numpy.log() function
    new_prob_list = []
    for j in prob_list:
        if np.array(j) > 1.e-30:
            new_prob_list.append(j)
    s_list = [-p*np.log(p) for p in new_prob_list]
    s = sum(s_list) # entropy
    return s

def free_energy(unit_list,params,
                Hamiltonian,L,
                state_vcl,state_vcr,
                H_bvecl,H_bvecr,T):
    """
    Free energy function
    """
    TN = contractions(unit_list,params,Hamiltonian,L,state_vcl,state_vcr,H_bvecl,H_bvecr,T)
    l_uc = len(unit_list)
    E = TN.contract(all, optimize='auto-hq')
    p_list = [prob_list(unit_list[j],params[j],T) for j in range(l_uc)]
    s_list = [entropy(p) for p in p_list]
    F = E - T*sum(s_list)/l_uc
    return F
