

# Sample code for Transverse Field Ising Model (TFIM) with bond dimension 2

import numpy as np
import random
from scipy.optimize import minimize
# import Circuit from circuit_qubit 
# import free_energy from thermal_state

# parameters of model 
d = 2
chimax = 2
# J = TBD
# g = TBD
# L = TBD
# N = TBD
# T = TBD

# circuit construction
c = Circuit([("qubit", "p", d), ("qubit", "b", chimax)])

def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

# one qubit rotation
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_x)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_x)
c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_z)

# XX rotation
c.add_gate("XX", qids=[0, 1], n_params = 1, fn = qub_two)

# YY rotation 
c.add_gate("YY", qids=[0, 1], n_params = 1, fn = qub_two)    

# ZZ rotation
c.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)

# one qubit rotation
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_x)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_x)
c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_z)

c.assemble()

# construction of Hamiltonian MPO  
def ising_mpo(J, g):
    """
    Unit-cell matrix product operator for Ising model Hamiltonian. 
    Based on TenPy (github.com/tenpy/tenpy/blob/master/toycodes/b_model.py)
    """
    # Pauli matrices and spin operators
    sigmax = np.array([[0., 1], [1, 0.]])
    sigmay = np.array([[0., -1j], [1j, 0.]])
    sigmaz = np.array([[1, 0.], [0., -1]])
    Sx, Sy, Sz = 0.5*sigmax, 0.5*sigmay, 0.5*sigmaz
    id = np.eye(2)
    
    # structure of Ising model MPO unit cell
    H = np.zeros((3, 3, d, d), dtype=np.float)
    H[0, 0] = H[2, 2] = id
    H[0, 1] = Sx
    H[0, 2] = -g * Sz
    H[1, 2] = -J * Sx
    return H

# original index ordering: virutal left, virtual right, physical out, physical in
H_mat1 = np.swapaxes(ising_mpo(J,g),0,2) # changing axis ordering to: p_out, b_out, b_in, p_in 
H_mat2 = np.swapaxes(H_mat1,2,3) # changing axis ordering to: p_out, b_out, p_in, b_in
chi_H = H_mat2[0,:,0,0].size # size of Hamiltonian bond leg dimension
# Hamiltonian MPO boundary vectors
# H_bvecl = TBD
# H_bvecr = TBD
bdry_vecs2 = [H_bvecl,H_bvecr]

# defining state and corresponding free energy function
rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi, size=c.n_params)
# state boundary vectors
# state_bvecl = TBD
# state_bvecr = TBD
bdry_vecs1 = [state_bvecl,state_bvecr]
# prob_list (for density-matrix option) = TBD (set to None by default)

def state_free_energy(params, circuit, state_type):
    """
    params: Parameters of circuit structure.       
    circuit: Holographic-based circuit structure.
    state_type: One of "density_matrix" or "random_state" options.       
    """   
    state = thermal_state.free_energy(circuit,params,state_type,L,H_mat2,T,chi_H,None,bdry_vecs1,bdry_vecs2)  
    return state

# optimization 
result1 = minimize(state_free_energy,x0=params,args=(c,'density_matrix'),method= TBD,options={'disp': True})
sweet_spot1 = result1.x
result2 = minimize(state_free_energy,args=(c,'random_state'),x0=params,method= TBD,options={'disp': True})
sweet_spot2 = result2.x
F_density = state_free_energy(sweet_spot1,c,'density_matrix')
F_random = state_free_energy(sweet_spot2,c,'random_state')

# theoretical predictions (applications of Jordan-Wigner transformations):
def theory_free_energy(case_type, N, J, g, T):
    """
    case_type: One of "ferromagnetic" or "anti-ferromagnetic" options.
    N : total number of lattice sites (must be even/odd for ferromagnetic/anti-ferromagnetic case).
    For more detail, see  Y. He and H. Guo, J. Stat. Mech. (2017) 093101.
    """
    a_list = [] 
    p_list = [] 
    E = lambda J,g,k: np.sqrt((J/2)**2 + g**2 + J*g*np.cos(k)) # dispersion relation
    
    # for ferromagnetic TFIM case:
    if case_type == 'ferromagnetic':
        
        # checking whether N is even
        if N % 2 != 0:
            raise ValueError('for ferromagnetic case, N must be even')
        
        # anti-periodic boundary condition (ABC) k-values
        for j in range(1,N):
            while ((2*j-1)*(np.pi)/N) <= ((N-1)*(np.pi)/N):
                a_list.append((2*j-1)*(np.pi)/N)
                a_list.append(-(2*j-1)*(np.pi)/N)
                break
                
        # periodic boundary condition (PBC) k-values
        for j in range(1,N):
            while (2*j)*(np.pi)/N <= ((N-2)*(np.pi)/N):
                p_list.append((2*j)*(np.pi)/N)
                p_list.append(-(2*j)*(np.pi)/N)
                break
        p_list.append(0)
        p_list.append(np.pi)
        
        # calculations of partition functions for PBC (Zp1 & Zp2) and APBC (Za1 & Za2) contributions
        for k1,k2 in zip(p_list,a_list):
            Zp1,Zp2,Za1,Za2, = 1,1,1,1
            x1 = E(J,g,k1)/(2*T)
            x2 = E(J,g,k2)/(2*T)  
            
            # partition functions:
            Za1 = Za1*(np.exp(x2) + np.exp(-x2))
            Zp1 = Zp1*(np.exp(x1) + np.exp(-x1))
            Za2 = Za2*(np.exp(x2) - np.exp(-x2))
            Zp2 = Zp2*(np.exp(x1) - np.exp(-x1))
            
        # total partition function for ferromagnetic case    
        Z_FM = (1/2)*(Za1 + Za2 + Zp1 - (np.sign(g - J/2))*Zp2) 
        F = -T*np.log(Z_FM) # free energy (ferromagnetic case)
    
    # for anti-ferromagnetic TFIM case:
    elif case_type == 'anti-ferromagnetic':
        
        # checking whether N is odd
        if N % 2 == 0:
            raise ValueError('for anti-ferromagnetic case, N must be odd')
        
        # anti-periodic boundary condition (ABC) k-values
        for j in range(1,N):
            while ((2*j-1)*(np.pi)/N) <= ((N-2)*(np.pi)/N):
                a_list.append((2*j-1)*(np.pi)/N)
                a_list.append(-(2*j-1)*(np.pi)/N)
                break
        a_list.append(np.pi)       
        
        # periodic boundary condition (PBC) k-values
        for j in range(1,N):
            while (2*j)*(np.pi)/N <= ((N-1)*(np.pi)/N):
                p_list.append((2*j)*(np.pi)/N)
                p_list.append(-(2*j)*(np.pi)/N)
                break
        p_list.append(0)
               
        # calculations of partition functions for PBC (Zp1 & Zp2) and APBC (Za1 & Za2) contributions
        for k1,k2 in zip(p_list,a_list):
            Zp1,Zp2,Za1,Za2, = 1,1,1,1
            x1 = E(J,g,k1)/(2*T)
            x2 = E(J,g,k2)/(2*T)
            
            # partition functions:
            Za1 = Za1*(np.exp(x2) + np.exp(-x2))
            Zp1 = Zp1*(np.exp(x1) + np.exp(-x1))
            Za2 = Za2*(np.exp(x2) - np.exp(-x2))
            Zp2 = Zp2*(np.exp(x1) - np.exp(-x1))
        
        # total partition function for anti-ferromagnetic case
        Z_AFM = (1/2)*(Za1 - Za2 + Zp1 + (np.sign(g - abs(J)/2))*Zp2) 
        F = -T*np.log(Z_AFM) # free energy (anti-ferromagnetic case)
    return F

F_theory = theory_free_energy(TBD,N,J,g,T)
        
print("Density matrix method: F_density = {:.8}".format(F_density))
print("Random state method: F_random = {:.8}".format(F_random))
print("Theory: F_theory = {:.8}".format(F_theory))
