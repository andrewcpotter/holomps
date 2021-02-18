

# Sample code for Transverse Field Ising Model (TFIM) 
# thermal density matrix method

import numpy as np
import random
from scipy.optimize import minimize
# import Circuit from circuit_qubit 
# import free_energy from thermal_state

# parameters of model 
d = 2
chimax = 2
J = TBD
g = TBD
L = TBD
l_uc = TBD
T = TBD
bd = TBD

# circuit construction
def circuit(bd):
    """
    bd: bond dimension
    """
    # for bond dimension 2:
    if bd == 2:
        
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
    
    # for bond dimension 4:
    elif bd == 4:
        
        def qub_x(params): return (np.pi/2, 0, params[0])
        def qub_y(params): return (np.pi/2, np.pi/2, params[0])
        def qub_z(params): return (0, 0, params[0])
        def qub_two(params): return (params[0])
        #def qub_Rxy(params): return (params[0], params[1])
        #def qub_Rz(params): return (params[0])

        c = Circuit([("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])

        # first entangle the physical qubit with the first bond qubit
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

        # then do the same for the physical qubit and the second bond qubit

        # one qubit rotation
        c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_x)
        c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_y)
        c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

        c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_x)
        c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_y)
        c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_z)

        # XX rotation
        c.add_gate("XX", qids=[0, 2], n_params = 1, fn = qub_two)

        # YY rotation 
        c.add_gate("YY", qids=[0, 2], n_params = 1, fn = qub_two)    

        # ZZ rotation
        c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = qub_two)

        # one qubit rotation
        c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_x)
        c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_y)
        c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

        c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_x)
        c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_y)
        c.add_gate("rotation", qids = [2], n_params = 1, fn = qub_z)

        c.assemble()   
    return c

c = circuit(bd)

# free energy
N = L * l_uc
def free_energy(params, circuit, N, T, Hamiltonian, chi_H, bdry_vecs1, bdry_vecs2):
    """
    params: Parameters of circuit structure.       
    circuit: Holographic-based circuit structure.
    N: Number of sites in the main network chain (= L * l_uc)
    T: Tempreture.
    Hamiltonian: The unit cell of the Hamiltonian MPO of model.  
    chi_H: Bond leg dimension for Hamiltonian MPO structure.
    bdry_vecs1 and bdry_vecs2: List of boundary vectors for state and Hamiltonian networks.
    """   
    F = thermal_state.free_energy(circuit,params,'density_matrix',N,Hamiltonian,T,chi_H,bdry_vecs1,bdry_vecs2,None)  
    return F


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

Hamiltonian = np.swapaxes(np.swapaxes(ising_mpo(J,g),0,2),2,3) # changing axis ordering to: p_out, b_out, p_in, b_in 
chi_H = Hamiltonian[0,:,0,0].size # size of Hamiltonian bond leg dimension
# Hamiltonian MPO boundary vectors
H_bvecl = TBD
H_bvecr = TBD
bdry_vecs2 = [H_bvecl,H_bvecr]

# defining state properties
rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi,size=c.n_params + N*d)
# state boundary vectors
state_bvecl = TBD
state_bvecr = TBD
bdry_vecs1 = [state_bvecl,state_bvecr]

# optimization 
result = minimize(free_energy,x0=params,args=(c,N,T,Hamiltonian,chi_H,bdry_vecs1,bdry_vecs2),method = TBD,options={'disp': True})
sweet_spot = result.x
F_density = free_energy(sweet_spot,c,N,T,Hamiltonian,chi_H,bdry_vecs1,bdry_vecs2)

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
        
        if N % 2 != 0: # checking whether N is even
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
        p1,p2,a1,a2 = [],[],[],[]
        for kp,ka in zip(p_list,a_list):         
            xp = E(J,g,kp)/(2*T)
            xa = E(J,g,ka)/(2*T)  
            p1.append(np.exp(xp) + np.exp(-xp))
            p2.append(np.exp(xp) - np.exp(-xp))
            a1.append(np.exp(xa) + np.exp(-xa))
            a2.append(np.exp(xa) - np.exp(-xa))
            
        # partition functions
        Zp1 = np.prod(p1)
        Zp2 = np.prod(p2)
        Za1 = np.prod(a1)
        Za2 = np.prod(a2)
            
        # total partition function for ferromagnetic case    
        Z_FM = (1/2)*(Za1 + Za2 + Zp1 - (np.sign(g - J/2))*Zp2) 
        F = -(T/N)*np.log(Z_FM) # free energy (ferromagnetic case)
    
    # for anti-ferromagnetic TFIM case:
    elif case_type == 'anti-ferromagnetic':    
        
        if N % 2 == 0: # checking whether N is odd
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
        p1,p2,a1,a2 = [],[],[],[]
        for kp,ka in zip(p_list,a_list):         
            xp = E(J,g,kp)/(2*T)
            xa = E(J,g,ka)/(2*T)  
            p1.append(np.exp(xp) + np.exp(-xp))
            p2.append(np.exp(xp) - np.exp(-xp))
            a1.append(np.exp(xa) + np.exp(-xa))
            a2.append(np.exp(xa) - np.exp(-xa))
            
        # partition functions
        Zp1 = np.prod(p1)
        Zp2 = np.prod(p2)
        Za1 = np.prod(a1)
        Za2 = np.prod(a2)
        
        # total partition function for anti-ferromagnetic case
        Z_AFM = (1/2)*(Za1 - Za2 + Zp1 + (np.sign(g - abs(J)/2))*Zp2) 
        F = -(T/N)*np.log(Z_AFM) # free energy (anti-ferromagnetic case)   
    else:
        raise ValueError('only one of "ferromagnetic" or "anti-ferromagnetic" options')
    return F

F_theory = theory_free_energy(TBD,N,J,g,T)
        
print("Density matrix method: F_density = {:.8}".format(F_density))
print("Theory: F_theory = {:.8}".format(F_theory))