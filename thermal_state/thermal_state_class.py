

import numpy as np
import random

class thermal_state(object):
    
    """
    Represents thermal states as Density Matrix Product Operator (DMPO)
    or (random) hologrphic Matrix Product State (random-holoMPS).      
    """
    def __init__(self, tensor, L):
        """
        Parameters
        --------------
        L: int
            Length (number) of repetitions of the unit cell in the main network chain.
        tensor: numpy.array
            Bulk rank-4 tensors of the main chain.
            tensor index ordering: physical-out, bond-out, physical-in, bond-in
            (with "in/out" referring to the right canonical form ordering)               
        """
        
        self.L = L
        self.tensor = tensor
        # tensor dimensions (consistent with rank-4 structure)
        self.d = tensor[:,0,0,0].size # physical leg dimension (assumes rank-4 structure)
        self.chi = tensor[0,:,0,0].size # bond leg dimension (assumes rank-4 structure)
        
    def network_from_cells(self, params, network_type, L, bdry_vecs=[None,None]):      
        """
        Returns network of finite random holo-Matrix Prodcut State (random_holoMPS), finite 
        holo-MPS (circuit_MPS), finite holo-Matrix Prodcut Operator (circuit_MPO), or MPO
        of a given tensor.
        --------------
        Inputs:
          --the input assumes either circuit w/ parameters or rank-4 numpy.array tensors--       
          network_type: str
             One of "random_MPS", "circuit_MPS", "circuit_MPO", or "MPO" options.
          L: int
             Length (number) of repetitions of unit cell in the main network chain.   
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default which gives left and right boundary vectors = |0>)       
        Note:
          -For random_MPS, circuit_MPS, and circuit_MPO, the original circuit with parameters
           must be inserted as args. In this case, the returned list of bulk tensors includes 
           rank-3 numpy.arrays for random_MPS/circuit_MPS and rank-4 numpy.arrays for circuit_MPO 
           (index ordering consistent with holographic-based simulations).
          -For holoMPS_based structures, the index ordering is: site, physical, bond-out, bond-in 
           (with "in/out" refer to right canonical form ordering).
          -For MPO, the unit cell tensor of MPO network must be inserted as arg (e.g. Hamiltonian 
           unit cell). Bulk tensors must be rank-4 numpy.array (consistent with final structure
           of MPO) with index ordering: physical-out, bond-out, physical-in, bond-in.
        """
        
        # for circuit_based structures:
        # both the circuit and params must be included
        if network_type == 'random_MPS' or network_type == 'circuit_MPS' or network_type == 'circuit_MPO':       
            unitary = self.get_tensor(params)
            
            # if network_type is set to random_MPS:
            if network_type == 'random_MPS':
                
                # tensor dimensions
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) # change to MPS_based structure
                d_r = tensor[:,0,0].size # physical leg dimension (for random state)
                chi_r = tensor[0,:,0].size # bond leg dimension (for random state)
            
                # random selections of each site
                tensor_list = [np.swapaxes(unitary[:,:,j,:],1,2) for j in range(d_r)]  
                tensor_list1 = [random.choice(tensor_list) for j in range(L)]
            
                # testing boundary conditions
                bdry = []
                for j in range(2):
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_r:
                        bdry.append(bdry_vecs[j])  
                    elif np.array(bdry_vecs[j] == None).all():
                        # if bdry_vec not specified, set to (1,0,0,0...)
                        bdry += [np.zeros(chi_r)]
                        bdry[j][0]=1
                    else:
                        if bdry_vecs[j].size != chi_r:
                            raise ValueError('boundary vector different size than bulk tensors')
                        bdry_vecs += [bdry_vecs[j]]
            
                M = [[bdry[0]],tensor_list1,[bdry[1]]]
            
            # if network_type is set to holoMPS:
            elif network_type == 'circuit_MPS':
                
                # tensor dimensions
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) # change to MPS_based structure
                d_c = tensor[:,0,0].size # physical leg dimension (for holoMPS)
                chi_c = tensor[0,:,0].size # bond leg dimension (for holoMPS)
            
                # bulk tensors of holoMPS structure
                tensors = L*[unitary]    
                tensor_list1 = [np.swapaxes(unitary[:,:,0,:],1,2) for unitary in tensors]
                
                # testing boundary conditions
                bdry = []
                for j in range(2):
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_c:
                        bdry.append(bdry_vecs[j])  
                    elif np.array(bdry_vecs[j] == None).all():
                        # if bdry_vec not specified, set to (1,0,0,0...)
                        bdry += [np.zeros(chi_c)]
                        bdry[j][0]=1
                    else:
                        if bdry_vecs[j].size != chi_c:
                            raise ValueError('boundary vector different size than bulk tensors')
                        bdry_vecs += [bdry_vecs[j]]
            
                M = [[bdry[0]],tensor_list1,[bdry[1]]]
            
            # if network_type is set to circuit_MPO 
            # this option assumes original, circuit_based MPO structure (e.g. holoMPO)
            elif network_type == 'circuit_MPO':
                
                # tensor dimensions (consistent with rank-4 structure)
                # index ordering consistent with holographic_based MPO structures
                d_MPO = unitary[:,0,0,0].size # physical leg dimension (for MPO)
                chi_MPO = unitary[0,:,0,0].size # bond leg dimension (for MPO)
                tensor_list1 = L*[unitary]
            
                # testing boundary conditions
                bdry = []
                for j in range(2):
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_MPO:
                        bdry.append(bdry_vecs[j])
                    elif np.array(bdry_vecs[j] == None).all():
                        # if bdry_vec not specified, set to (1,0,0,0...)
                        bdry += [np.zeros(chi_MPO)]
                        bdry[j][0]=1
                    else:
                        if bdry_vecs[j].size != chi_MPO:
                            raise ValueError('boundary vector different size than bulk tensors')
                        bdry_vecs += [bdry_vecs[j]]
                       
                M = [[bdry[0]],tensor_list1,[bdry[1]]]
                
        
        # if network_type is set to MPO: 
        # this option assumes genuine MPO_based structures (e.g. Hamiltonian MPO)  
        # only the bulk tensors of the main chain must be included (w/ params=None)
        elif network_type == 'MPO':
            
            # tensor dimensions (consistent with rank-4 structure)
            # index ordering consistent with holographic_based MPO structures
            d_MPO = self[:,0,0,0].size # physical leg dimension (for MPO)
            chi_MPO = self[0,:,0,0].size # bond leg dimension (for MPO)
            tensor_list1 = L*[self]
            
            # testing boundary conditions
            bdry = []
            for j in range(2):
                if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_MPO:
                    bdry.append(bdry_vecs[j])
                elif np.array(bdry_vecs[j] == None).all():
                    # if bdry_vec not specified, set to (1,0,0,0...)
                    bdry += [np.zeros(chi_MPO)]
                    bdry[j][0]=1
                else:
                    if bdry_vecs[j].size != chi_MPO:
                        raise ValueError('boundary vector different size than bulk tensors')
                    bdry_vecs += [bdry_vecs[j]]
                
            M = [[bdry[0]],tensor_list1,[bdry[1]]]
        
        return M

    def density_matrix(self, params, L, prob_list, bdry_vecs=[None,None]):      
        """
        Returns Density Matrix Product Operator (DMPO) of a tensor network.
        --------------
        Inputs:
          --circuit structure w/ parameters--
          L: int
             length (number) of repetitions of unit cell in the main network chain.
          prob_list: list 
             List of probabilities of each physical state.
             Length of prob_list should match the physical leg dimension.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default which gives left and right boundary vectors = |0>)
        """ 
        
        unitary = self.get_tensor(params)
        
        # tensor dimensions (consistent with rank-4 structure)
        # index ordering consistent with holographic_based MPO structures
        d = unitary[:,0,0,0].size # physical leg dimension
        chi = unitary[0,:,0,0].size # bond leg dimension
        if len(prob_list) != d:
            raise ValueError('length of probability list should match the physical dimension')      
        
        
        state1 = thermal_state.network_from_cells(self,params,'circuit_MPO',L,bdry_vecs) # constructing state as MPO
        # defining and constructing the dual state as MPO
        tensor_conj = unitary.conj()
        bdry_vecs_conj = [state1[0][0].conj(),state1[2][0].conj()] # boundary vectors of dual state  
        state2 = thermal_state.network_from_cells(tensor_conj,None,'MPO',L,bdry_vecs_conj)
        
        # constructing the probability network chain
        p_matrix = np.diag(prob_list)
        p_state = L*[p_matrix]     
        
        # density matrix main tensor network and state/prob_list/dual_state contractions 
        TN1 = [np.tensordot(state1[1][j],p_state[j],axes=[2,0]) for j in range(L)]
        TN2 = [np.tensordot(np.reshape(element,[d,chi,d,chi]), state2[1][k] ,axes=[2,2]) for element,k in zip(TN1,range(L))]
        
        # boundary contractions
        BvL = np.kron(state1[0][0],state2[0][0])
        BvR = np.kron(state1[2][0],state2[2][0])
        
        density_matrix = [[BvL],TN2,[BvR]]
        
        return density_matrix

    def transfer_matrix(self, MPO=None):
        """
        Returns transfer matrix of a given Matrix product State (MPS) structure
        (might also include a Matrix Prodcut Operator (MPO) between the states).
        --------------
        Inputs:
          --the input assumes thermal_state_class-based random-holoMPS, holoMPS, and MPO networks--
        Note:
          -If MPO is not inserted, the function computes the transfer matrix for the state 
           wave fucntion.
          -MPO (built by thermal_state.networks) has the same physical dimension as MPS.
        """
        
        # tensor dimensions (consistent with rank-3 structure)
        # index ordering consistent with holoMPS structure
        tensor = self[1][0]
        d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
        chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
        
        # transfer matrix for the wave function (without MPO inserted)
        if MPO == None:        
            t_mat_site = np.zeros([chi**2,chi**2],dtype=complex) # transfer matrix at each site
            t_mat = np.eye(chi**2,chi**2,dtype=complex) # total transfer matrix 
            
            # site contractions for state and its dual
            for j in range(len(self[1])):
                t_tensor =  np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                # reshaping into matrix
                t_mat_site = np.reshape(np.swapaxes(t_tensor,1,2),[chi**2,chi**2]) 
                t_mat = t_mat_site @ t_mat # t-mat accumulations                 
        
        # transfer matrix with MPO inserted
        else:
            chi_MPO = MPO[1][0][0,:,0,0].size # bond dimension of MPO
            t_mat_site = np.zeros([chi**2 * chi_MPO,chi**2 * chi_MPO]) # transfer matrix at each site
            t_mat = np.eye(chi**2 * chi_MPO,chi**2 * chi_MPO,dtype=complex) # total transfer matrix
            
            # site contractions for state/MPO/dual_state
            for j in range(len(self[1])):
                # contractions with state
                chi_s = chi_MPO * chi
                MPO_on_state = np.tensordot(MPO[1][j],self[1][j],axes=[2,0])
                MPO_on_state_reshape = np.reshape(np.swapaxes(MPO_on_state,2,3),(d,chi_s,chi_s))

                # constractions with dual state
                chi_tot = chi_s * chi # total bond dimension
                t_mat_site = np.reshape(np.swapaxes(np.tensordot(self[1][j].conj(),MPO_on_state_reshape,axes=[0,0]),1,2),[chi_tot,chi_tot])
                t_mat = t_mat_site @ t_mat # t-mat accumulations
        return t_mat

    def expectation_value(self, L, MPO=None):
        """
        Returns the result of <MPS|MPO|MPS> constraction.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based random-holoMPS, holoMPS, and MPO networks--
        Note:
          -The left boundary condition is set by holoMPS boundary vectors and the right 
           condition is averaged over.
          -If MPO is not inserted, the function computes the expectation value for the state 
           wave fucntion (<MPS|MPS>).
          -MPO (built by thermal_state.networks) has the same physical dimension as MPS.
        """
        
        # tensor dimensions (consistent with rank-3 structure)
        # index ordering consistent with holoMPS structure
        tensor = self[1][0]
        d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
        chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
        t_mat = np.linalg.matrix_power(thermal_state.transfer_matrix(self,MPO),L)
    
        # without MPO inserted
        if MPO == None:
            bvec_l = np.kron(self[0][0].conj(),self[0][0])
            if np.array(self[2][0]==None).all():
            # summing over right vectors if the right boundary condition is not specified 
                t_mat_on_rvec = np.reshape(t_mat @ bvec_l,[chi**2])
                rvec = np.reshape(np.eye(chi),[chi**2])
                expect_val = np.dot(rvec, t_mat_on_rvec)
            else:
                bvec_r = np.kron(self[2][0].conj(),self[2][0])
                expect_val = bvec_r.conj().T @ t_mat @ bvec_l
        
        # with MPO inserted
        else:
            bvec_l = np.kron(self[0][0].conj(),np.kron(MPO[0][0],self[0][0]))      
            if np.array(self[2][0]==None).all():
             # summing over right vectors if the right boundary condition is not specified
             # employ the specified right-boundary vector of the MPO.         
                chi_MPO = MPO[1][0][0,:,0,0].size # bond dimension of MPO        
                t_vleft = np.reshape((t_mat @ bvec_l),[chi,chi_MPO,chi]) # t_matrix on left vector
                MPO_rvec_contracted = np.reshape(np.tensordot(MPO[2][0],t_vleft,axes=[0,1]),[chi**2])
                rvec = np.reshape(np.eye(chi),[chi**2])
                expect_val = np.dot(rvec,MPO_rvec_contracted)
            else:
                bvec_r = np.kron(self[2][0].conj(),np.kron(MPO[2][0],self[2][0]))
                expect_val = bvec_r.conj().T @ t_mat @ bvec_l
        return (expect_val).real

    def free_energy(self, params, state_type, L, H_mat, T, prob_list=None, bdry_vecs1=[None,None], bdry_vecs2=[None,None]):
        """
        Returns the free energy of a density matrix/random-holoMPS state.
        --------------
        Inputs:
        --circuit structure w/ parameters--
        state_type: str
           One of "density_matrix" or "random_state" options.
        L: int 
           Length (number) of repetitions of unit cell in the main network chain.
        H_mat: numpy.array 
           The Hamiltonian of model. 
           Should have the same physical dimension as the state. 
        T: float
           Tempreture of system.
        prob_list (for "density_matrix" option): list 
             List of probabilities of each physical state.
             Length of prob_list should match the physical leg dimension.
        bdry_vecs1 and bdry_vecs2: list
            List of left (first element) and right (second element) boundary vectors for each network.
            (set to [None,None] by default which gives left and right boundary vectors = |0>)    
        """
        
        unitary = self.get_tensor(params)
        
        # free energy for density matrix state
        if state_type == 'density_matrix':
                 
            # tensor dimensions (consistent with rank-4 structure)
            # index ordering consistent with holographic_based MPO structures
            d = unitary[:,0,0,0].size # physical leg dimension (assumes rank-4 structure)
            chi = unitary[0,:,0,0].size # bond leg dimension (assumes rank-4 structure)
            if len(prob_list) != d:
                raise ValueError('length of probability list should match the physical dimension')
            
            # constructing density matrix and Hamiltonian (as MPO)
            density_mat = thermal_state.density_matrix(self,params,L,prob_list,bdry_vecs1)
            Hamiltonian = thermal_state.network_from_cells(H_mat,None,'MPO',L,bdry_vecs2)
            
            # the main tensor network and boundary vectors' constrcution
            TN = [np.tensordot(Hamiltonian[1][j],density_mat[1][j],axes=[2,0]) for j in range(L)]
            BL = np.kron(density_mat[0][0],Hamiltonian[0][0])
            BR = np.kron(density_mat[2][0],Hamiltonian[2][0]) 
            
            # network contractions
            W0 = TN[0]
            for j in range(1,len(TN)):
                # reshaping and contracting tensors to matrix forms (consistent with physical/bond dimensions) 
                W0 = np.tensordot(np.reshape(W0,[d**2,chi**2,d**2,chi**2]),np.reshape(TN[j],[d**2,chi**2,d**2,chi**2]))
            # contraction with boundary conditions          
            W1 = np.outer(BL,W0)
            W2 = np.outer(W1,BR)
            
            # entropy calculation 
            S = 0
            S_list = [-p*np.log(p) for p in prob_list]
            for j in range(len(S_list)):
                S = S + S_list[j]
            
            E = (np.trace(W2)).real
            F = E - T*S # Helmholtz free energy
        
        # free energy for random-holoMPS state
        elif state_type == 'random_state':
            
            # constructing random state and Hamiltonian (as MPO)
            random_state = thermal_state.network_from_cells(self,params,'random_MPS',L,bdry_vecs1)
            Hamiltonian = thermal_state.network_from_cells(H_mat,None,'MPO',L,bdry_vecs2)
            
            # tensor dimensions (consistent with rank-3 structure)
            # index ordering consistent with holoMPS structure
            d = random_state[1][0][:,0,0].size # physical leg dimension (for holoMPS) 
            chi = random_state[1][0][0,:,0].size # bond leg dimension (for holoMPS)
            
            # calculation of prababilities of each physical state
            count = [0]*d
            tensor_list = [np.swapaxes(unitary[:,:,j,:],1,2) for j in range(d)]
            # counting the number of individual physical tensors in tensor_list
            for tensor in random_state[1]:
                for j in range(len(tensor_list)):
                    if (tensor == tensor_list[j]).all():
                        count[j] += 1  
            prob_list = [c/L for c in count]
            
            # entropy calculation 
            S = 0
            S_list = [-p*np.log(p) for p in prob_list]
            for j in range(len(S_list)):
                S = S + S_list[j]
                
            E = thermal_state.expectation_value(random_state,L,Hamiltonian)
            F = E - T*S # Helmholtz free energy      
        return F
