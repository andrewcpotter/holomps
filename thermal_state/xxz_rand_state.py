

# Sample code for XXZ Heisenberg Chain
# thermal holoMPS method

import numpy as np
import random
from scipy.optimize import minimize
# import Circuit from circuit_qubit 
# import free_energy from thermal_state

# parameters of model 
d = 2
chimax = 2
J = TBD
Delta = TBD
hz = TBD
L = TBD
l_uc = TBD
T = TBD
bd = TBD
method = TBD

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
def free_energy(params, circuit, N, T, Hamiltonian, bdry_vecs1, bdry_vecs2, method):
    """
    params: Parameters of circuit structure.       
    circuit: Holographic-based circuit structure.
    N: Number of sites in the main network chain (= L * l_uc)
    T: Tempreture.
    Hamiltonian: The unit cell of the Hamiltonian MPO of model.  
    bdry_vecs1 and bdry_vecs2: List of boundary vectors for state and Hamiltonian networks.
    method: one of "thermal_state_class" or "tenpy" options.
    """
    if method == 'thermal_state_class':      
        chi_H = Hamiltonian[0,:,0,0].size # size of Hamiltonian bond leg dimension
        F = thermal_state.free_energy(circuit,params,'random_state',N,Hamiltonian,T,chi_H,bdry_vecs1,bdry_vecs2,'thermal_state_class')  
    elif method == 'tenpy':
        F = thermal_state.free_energy(circuit,params,'random_state',N,Hamiltonian,T,None,None,None,'tenpy')      
    else:
        raise ValueError('only one of "thermal_state_class" or "tenpy" options')        
    return F

# construction of Hamiltonian MPO  
def xxz_mpo(J, Delta, hz, method, N=None):
    """
    method: one of "thermal_state_class" or "tenpy" options.
    N: number of sites (for TenPy option).
    """    
    if method == 'thermal_state_class':
        # Pauli matrices and spin operators
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        Sx, Sy, Sz = 0.5*sigmax, 0.5*sigmay, 0.5*sigmaz
        Sp, Sm = Sx + 1j*Sy, Sx - 1j*Sy
        id = np.eye(2)   
        # structure of XXZ MPO unit-cell
        H0 = np.zeros((5, 5, d, d), dtype=np.float)
        H0[0, 0] = H0[4, 4] = id
        H0[0, 1] = Sp
        H0[0, 2] = Sm
        H0[0, 3] = Sz
        H0[0, 4] = -hz * Sz
        H0[1, 4] = 0.5 * J * Sm
        H0[2, 4] = 0.5 * J * Sp
        H0[3, 4] =  J * Delta * Sz
        H = np.swapaxes(np.swapaxes(H0,0,2),2,3) # changing axis ordering to: p_out, b_out, p_in, b_in
        
    elif method == 'tenpy':
        site = SpinHalfSite(None)
        Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
        H_bulk = [[Id, Sp, Sm, Sz, -hz * Sz],
                  [None, None, None, None, 0.5 * J * Sm],
                  [None, None, None, None, 0.5 * J * Sp],
                  [None, None, None, None, J * Delta * Sz],
                  [None, None, None, None, Id]]
        
        H_first = [H_bulk[0]] # first row
        H_last = [[row[-1]] for row in H_bulk] # last column
        H = [H_first] + [H_bulk]*(N - 2) + [H_last]   
    else:
        raise ValueError('only one of "thermal_state_class" or "tenpy" options')
    return H

Hamiltonian = xxz_mpo(J,Delta,hz,method,N)  
# Hamiltonian MPO boundary vectors (only for thermal_state_class method)
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
result = minimize(free_energy,x0=params,args=(c,N,T,Hamiltonian,bdry_vecs1,bdry_vecs2,method),method = TBD,options={'disp': True})
sweet_spot = result.x
F_random = free_energy(sweet_spot,c,N,T,Hamiltonian,bdry_vecs1,bdry_vecs2,method)     
print("Random state method: F_random = {:.8}".format(F_random))
