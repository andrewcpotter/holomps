

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

# defining state and Hamiltonian 
rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi, size=c.n_params)

# Pauli matrices and spin operators
sigmax = np.array([[0., 1], [1, 0.]])
sigmay = np.array([[0., -1j], [1j, 0.]])
sigmaz = np.array([[1, 0.], [0., -1]])
id = np.eye(2)
H = np.reshape(-J * np.kron(sigmax, sigmax) - g * np.kron(sigmaz, id),[d, d, d, d])
# prob_list = TBD (should be a list of probabilities for each site (e.g. [p1,...,pn]))

# method I. direct calculation
# F_density = thermal_state.free_energy(c,params,'density_matrix',L_d,H,T,prob_list,[None,None],[None,None])
# F_random = thermal_state.free_energy(c,params,'random_state',L_d,H,T,prob_list,[None,None],[None,None])

# method II. nelder-mead minimization method
# def density_free_energy(params,circuit):
#     state = thermal_state.free_energy(circuit,params,'density_matrix',L_d,H,T,prob_list,[None,None],[None,None])
#     return state

# def random_free_energy(params,circuit): 
#    state = thermal_state.free_energy(c,params,'random_state',L_r,H,T,None,[None,None],[None,None])
#    return state

# result1 = minimize(density_free_energy,args=(c),x0=params,method='nelder-mead')
# sweet_spot1 = result1.x
# result2 = minimize(random_free_energy,args=(c),x0=params,method='nelder-mead')
# sweet_spot2 = result2.x
# F_density = density_free_energy(sweet_spot1,c)
# F_random = random_free_energy(sweet_spot2,c)

# theoretical predictions for ferromagnetic case (applications of Jordan-Wigner transformations)
# for more details, see  Y. He and H. Guo, J. Stat. Mech. (2017) 093101.
# total number of sites must be even for ferromagnetic case.

# k-values for periodic and anti-periodic boundary conditions (PBC and APBC)
a_list = [] # for APBC
p_list = [] # for PBC
for j in range(1,L):
    while ((2*j-1)*(np.pi)/L) <= ((L-1)*(np.pi)/L):
        a_list.append((2*j-1)*(np.pi)/L)
        a_list.append(-(2*j-1)*(np.pi)/L)
        break
for j in range(1,L):
    while (2*j)*(np.pi)/L <= ((L-2)*(np.pi)/L):
        p_list.append((2*j)*(np.pi)/L)
        p_list.append(-(2*j)*(np.pi)/L)
        break
p_list.append(0)
p_list.append(np.pi)

E = lambda J,g,k: np.sqrt((J/2)**2 + g**2 + J*g*np.cos(k)) # dispersion relation

# calculations of partition functions for PBC (Zp1 & Zp2) and APBC (Za1 & Za2) contributions
for k1,k2 in zip(p_list,a_list):
    Zp1,Zp2,Za1,Za2, = 1,1,1,1
    x1 = E(J,g,k1)/(2*T)
    x2 = E(J,g,k2)/(2*T)
    
    Za1 = Za1*(np.exp(x2) + np.exp(-x2))
    Zp1 = Zp1*(np.exp(x1) + np.exp(-x1))
    Za2 = Za2*(np.exp(x2) - np.exp(-x2))
    Zp2 = Zp2*(np.exp(x1) - np.exp(-x1))
    
ZFM = (1/2)*(Za1 + Za2 + Zp1 - (np.sign(g - J/2))*Zp2) # total partition function for ferromagnetic case
F_theory = -T*np.log(ZFM) # free energy

print("Density matrix method: F_density = {:.8}".format(F_density))
print("Random state method: F_random = {:.8}".format(F_random))
print("Theory: F_theory = {:.8}".format(F_theory))
