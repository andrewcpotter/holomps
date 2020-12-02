
# Test for free energy calculations of Transverse Field Ising Model (TFIM)

import numpy as np
import random
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from circuit_qubit import Circuit

# Parameters of model
d = 2
chimax = 2
# J = TBD 
# g = TBD
# L = TBD
# T = TBD

# Circuit construction
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

# Defining state and Hamiltonian for density matrix method
rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi, size=c.n_params)

unitary = c.get_tensor(params) 
sigmax = np.array([[0., 1], [1, 0.]])
sigmay = np.array([[0., -1j], [1j, 0.]])
sigmaz = np.array([[1, 0.], [0., -1]])
sx, sy, sz = 0.5*sigmax, 0.5*sigmay, 0.5*sigmaz
id = np.eye(2)
# prob_list = TBD (should be a list of probabilities for each site (e.g. [p1,...,pn]))
density_matrix = thermal_state.density_matrix(unitary,L,prob_list,bdry_vecs = [None,None])
H = np.reshape(-J * np.kron(sx, sx) - g * np.kron(sz, id),[d, d, d, d])

F_density = thermal_state.free_energy(unitary,'density matrix',L,H,params,T,prob_list,[None,None],[None,None])

# Defining state and Hamiltonian for random holoMPS method
def ising_mpo(J, g, L):
    site = SpinHalfSite(None)
    Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
    Sx= Sp + Sm
    # confirmed from TeNPy docs, lines below directly copied from docs
    W_bulk = [[Id,Sx,g*Sz],[None,None,-J*Sx],[None,None,Id]] 
    W_first = [W_bulk[0]]  # first row
    W_last = [[row[-1]] for row in W_bulk]  # last column
    Ws = [W_first] + [W_bulk]*(L-2) + [W_last]
    H = MPO.from_grids([site]*L, Ws, bc='finite',IdL=0, IdR=-1) # (probably leave the IdL,IdR)
    return H

# L2 = TBD 
random_state = thermal_state.random(c,params,L2)
H_mat = ising_mpo(J, g, L2)
F_random = thermal_state.free_energy(c,'random state', L2, H_mat, params, T, prob_list=None, bdry_vecs1=None, bdry_vecs2=None)


# Theoretical predictions for ferromagnetic case (applications of Jordan-Wigner transformations)
# for more details see https://arxiv.org/pdf/1707.02400.pdf
# k-values (momenta) for periodic and anti-periodic boundary conditions (PBC and APBC)
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

def E(J,g,k):
    return np.sqrt((J/2)**2 + g**2 + J*g*np.cos(k)) # the dispersion relation

# calculations of partition functions for PBC (Zp1 & Zp2) and APBC (Za1 & Za2) contributions
for k1,k2 in zip(p_list,a_list):
    Zp1,Zp2,Za1,Za2, = 1,1,1,1
    x1 = E(J,g,k1)/(2*T)
    x2 = E(J,g,k2)/(2*T)
    
    Za1 = Za1*(np.exp(x2) + np.exp(-x2))
    Zp1 = Zp1*(np.exp(x1) + np.exp(-x1))
    Za2 = Za2*(np.exp(x2) - np.exp(-x2))
    Zp2 = Zp2*(np.exp(x1) - np.exp(-x1))
ZFM = (1/2)*(Za1 + Za2 + Zp1 - (np.sign(g - J/2))*Zp2) #total partition function for ferromagnetic case

F_theory = -T*np.log(ZFM) 

print("Density matrix method: F_density = {:.8}".format(F_density))
print("Random method: F_random = {:.8}".format(F_random))
print("Theory: F_theory = {:.8}".format(F_theory))
