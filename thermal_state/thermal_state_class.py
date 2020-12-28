
import numpy as np
import random
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS


class thermal_state(object):
    
    """
    Represents thermal states as Density Matrix Product Operator (DMPO)
    or (random) hologrphic Matrix Product State (random holoMPS).      
    """
    def __init__(self, tensor, L = np.inf):
        """
        Parameters
        --------------
        L: int
            Length (number) of repetitions of the unit cell in the main chain.
            (set to infinity by default for infinite chain)
        tensor: numpy.array
            Bulk rank-4 tensors of the main chain.
            tensor index ordering: physical-out, bond-out, physical-in, bond-in
            (with "in/out" referring to the right canonical form ordering)               
        """
        self.L = L
        self.tensor = tensor
        self.l_uc = len(tensor) # tensor dimension
        self.d = tensor[:,0,0,0].size # physical leg dimension
        self.chi = tensor[0,0,:,0].size # bond leg dimension
        
    def random(self, params, L): 
        """
        Returns the random holographic MPS of a given holographic circuit.
        --------------
        Inputs:
        params: numpy.array
            A vector of parameters for the gates.
        L: int
           Length (number) of repetitions of unit cell in the main chain.         
        output:
        A two-element list where the first element includes the list of 
        probabilities of each site in the MPS, and the second element 
        consists of the constructed state itself.
        """              
        tensor = self.get_tensor(params)
        d = tensor[:,0,0,0].size # physical leg dimension
        # random selection of each site
        tensor_list1 = [np.swapaxes(tensor[:,:,j,:],1,2) for j in range(d)]
        tensor_list2 = []
        for j in range(L):
            tensor_list2.append(random.choice(tensor_list1))
        # fixing the boundary vectors    
        tensor_list2[0] = tensor_list2[0][:,0:1,:]
        tensor_list2[-1] = tensor_list2[-1][:,:,0:1]
        
        # calculation of prabability of each state
        tensor_list3 = []
        prob_list = []
        count = [0]*d
        for j in range(d):
            tensor_list3.append(np.swapaxes(tensor[:,:,j,:],1,2))   
        # counting the number of individual tensors in tensor_list2
        for tensors in tensor_list2:
            for k in range(d):
                if (tensor_list3[k] == tensors).all():
                    count[k] += 1    
        for c in count:
            prob_list.append(c/(L-2))
        
        # constructing the final state using TenPy's MPS class (MPS must be in canonical form) 
        site = SpinHalfSite(None)     
        random_state = MPS.from_Bflat([site]*L, tensor_list2, bc='finite', dtype=complex, form=None)    
        MPS.canonical_form_finite(random_state,renormalize=True,cutoff=0.0)
        
        R_list = [prob_list,random_state] 
        return R_list
    
    def MPO_from_cells(self, L, bdry_vecs = [None,None]):
        """
        Returns Matrix Product Operator (MPO) of a given tensor.
        --------------
        Inputs:
          --Bulk tensors of MPO must be rank-4 numpy.array--
          --index ordering: physical-out, bond-out, physical-in, bond-in--
          L: int
             Length (number) of repetitions of unit cell in the main chain.   
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default, which gives left and right boundary vectors = |0>)
        """
        tensor_list = L*[self]
        bdry = []
        chi = self[0,0,:,0].size # bond leg dimension
        
        # testing the boundary conditions        
        for j in range(2):
            if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                bdry.append(bdry_vecs[j])  
            elif np.array(bdry_vecs[j] == None).all():
                # if bdry_vec not specified, set to (1,0,0,0...)
                bdry += [np.zeros(chi)]
                bdry[j][0]=1
            else:
                if bdry_vecs[j].size != chi:
                    raise ValueError('left boundary vector different size than bulk tensors')
                bdry_vecs += [bdry_vecs[j]]
        M = [[bdry[0]],tensor_list,[bdry[1]]]
        return M

    def density_matrix(self, L, prob_list, bdry_vecs = [None,None]):
        """
        Returns Density Matrix Product Operator (DMPO) of a given tensor.
        --------------
        Inputs:
          --Bulk tensors of MPO must be rank-4 numpy.array--
          --index ordering: physical-out, bond-out, physical-in, bond-in--
          L: int
             length (number) of repetitions of unit cell in the main chain.
          prob_list: list 
             List of probabilities of each physical state.
             (length of the list should match the physical leg dimension).
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default which gives left and right boundary vectors = |0>)
        """   
        d = self[:,0,0,0].size # physical leg dimension
        if len(prob_list) != d:
            raise ValueError('length of probability list should match the physical dimension') 
        
        # constructing state as MPO
        state1 = thermal_state.MPO_from_cells(self,L,bdry_vecs) 
        # defining and constructing the dual state as MPO
        tensor_conj = self.conj()
        bdry_vecs_conj = [state1[0][0].conj(),state1[2][0].conj()] # boundary vectors of dual state  
        state2 = thermal_state.MPO_from_cells(tensor_conj,L,bdry_vecs_conj)
        
        # constructing the probability chain
        p_matrix = np.diag(prob_list)
        p_state = L*[p_matrix]     
        
        # the main chain's tensor network and contractions
        TN1 = [np.tensordot(state1[1][j],p_state[j],axes=[2,0]) for j in range(L)]
        # checking the size of the bond dimension of the main tensor (e.g. 2 and 4)
        if self[0,:,0,0].size == 2:
            TN2 = [np.tensordot(element,state2[1][k],axes=[2,2]) for element,k in zip(TN1,range(L))]
        elif self[0,:,0,0].size == 4:       
            TN2 = [np.tensordot(element,state2[1][k],axes=[2,1]) for element,k in zip(TN1,range(L))]
        
        # boundary contractions
        BvL = np.kron(state1[0][0],state2[0][0])
        BvR = np.kron(state1[2][0],state2[2][0])
        
        density_matrix = [[BvL],TN2,[BvR]]
        
        return density_matrix

    def free_energy(self, state_type, L, H_mat, params, T, prob_list, bdry_vecs1 = [None,None], bdry_vecs2 = [None,None]):
        """
        Returns the free energy for a given density matrix/circuit.
        --------------
        Inputs:
        state_type: str
           One of the "density matrix" or "random" options.
        L: int
           Length (number) of repetitions of unit cell in the main chain.
        H_mat: numpy.array or MPO class from tenpy.networks.mpo
           The Hamiltonian of the given model. If state_type == 'density matrix',
           H_mat is set as numpy.array. If state_type == 'random', H_mat is represented 
           by the MPO class of tenpy.networks.mpo.
        params (optional for 'density matrix'): numpy.array 
            A vector of parameters for the gates. 
        T: float
           Temperature of the system.
        prob_list (optional for 'random'): list 
             List of probabilities of each physical state.
             (length of the list should match the physical leg dimension).
        bdry_vecs1: list
            List of left (first element) and right (second element) boundary vectors (for the density matrix).
            (set to [None,None] by default which gives left and right boundary vectors = |0>)
        bdry_vecs2: list
            List of left (first element) and right (second element) boundary vectors (for the Hamiltonian).
            (set to [None,None] by default which gives left and right boundary vectors = |0>)    
        """
        if state_type == 'density matrix':
            d = self[:,0,0,0].size # physical leg dimension
            if len(prob_list) != d:
                raise ValueError('length of probability list should match the physical dimension')
            
            # calculations for the entropy of the chain (for density matrix mode)
            S = 0
            S_list = [-p*np.log(p) for p in prob_list]
            for j in range(len(S_list)):
                S = S + S_list[j]
            
            # constructing the density matrix and Hamiltonian (as MPO)     
            density_mat = thermal_state.density_matrix(self,L,prob_list,bdry_vecs1)
            Hamiltonian = thermal_state.MPO_from_cells(H_mat,L,bdry_vecs2)
            # the main tensor network and boundary vectors' constrcution
            TN = [np.tensordot(Hamiltonian[1][j],density_mat[1][j],axes=[2,0]) for j in range(L)]
            BL = np.kron(density_mat[0][0],Hamiltonian[0][0])
            BR = np.kron(density_mat[2][0],Hamiltonian[2][0])   
            # the network's contractions
            W0 = TN[0]
            for j in range(1,len(TN)):
                # checking the size of the bond dimension 
                if self[0,:,0,0].size == 2:
                    W0 = np.tensordot(W0,TN[j])
                elif self[0,:,0,0].size == 4:
                    W0 = np.tensordot(W0,TN[j],axes=[4,2])
            W1 = np.outer(BL,W0)
            W2 = np.outer(W1,BR)
            
            E = (np.trace(W2)).real
            F = E - T*S
      
        elif state_type == 'random state':
            state = thermal_state.random(self,params,L)
                     
            # calculations for the entropy of the chain (for random state mode)
            S = 0
            prob_list = state[0]
            S_list = [-p*np.log(p) for p in prob_list]
            for j in range(len(S_list)):
                S = S + S_list[j]
                
            # expectation value of energy from TenPy's MPO class
            E = (H_mat.expectation_value(state[1])).real
            F = E - T*S          
        return F
