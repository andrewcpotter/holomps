

import numpy as np
import random

class thermal_state(object):
    
    """
    Represents thermal states (in the forms of Density Matrix Product Operator (DMPO)
    and (random) hologrphic Matrix Product State (random-holoMPS)) and is used for
    finite-temperature simulations.   
    """
    
    def __init__(self, tensor, L):
        """
        Parameters
        --------------
        L: int
            Length (number) of repetitions of the unit cell in the main network chain.
        tensor: numpy.ndarray
            Bulk rank-4 tensors of the main chain.
            tensor index ordering: physical-out, bond-out, physical-in, bond-in
            (with "in/out" referring to the right canonical form ordering)               
        """
        
        self.L = L
        self.tensor = tensor
        # tensor dimensions (consistent with rank-4 structure)
        self.d = tensor[:,0,0,0].size # physical leg dimension (assumes rank-4 structures)
        self.chi = tensor[0,:,0,0].size # bond leg dimension (assumes rank-4 structures)
     
    def network_from_cells(self, network_type, L, chi_MPO=None, params=None, bdry_vecs=[None,None]):      
        """
        Returns network of finite random-holographic Matrix Product State (random-holoMPS), finite 
        holo-MPS, finite holographic Matrix Product Operator (holoMPO), or MPO of a given tensor.
        --------------
        Inputs:
          --the input assumes either circuit or rank-4 numpy.ndarray tensors--       
          network_type: str
             One of "random_state", "circuit_MPS", "circuit_MPO", or "MPO" options.
          L: int
             Length (number) of repetitions of unit cell in the main network chain. 
          chi_MPO: int
             Bond leg dimension for MPO-based structures. 
          params: numpy.ndarray
             Parameters of circuit structure.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default, which gives left and right boundary vectors = |0> 
            for MPO-based structures. For holoMPS-based structures, the default [None, None]
            would give left boundary = |0> while the right boundary traced over).        
        Note:
          -For random_state, circuit_MPS and circuit_MPO options, the original circuit with 
           parameters must be inserted as args. In this case, the returned list of bulk tensors
           includes rank-3 numpy.ndarray for random_state/circuit_MPS and rank-4 numpy.ndarray for
           circuit_MPO.
          -For holoMPS-based structures, the index ordering is: site, physical, bond-out, bond-in,
           and for holoMPO-based structures, the index ordering is: physical-out, bond-out,
           physical-in, bond-in (with "in/out" referring to right canonical form ordering).
          -For MPO, the unit cell tensor of MPO network must be inserted as arg (e.g. Hamiltonian 
           unit cell). Bulk tensors must be rank-4 numpy.ndarray (consistent with final structure
           of MPO) 
          -Tracing over right boundary for holoMPS-based structures is appropriate for 
           holographic/sequential-based simulations.   
        """
        
        # for circuit-based structures:
        # both circuit and params must be included
        if network_type == 'random_state' or network_type == 'circuit_MPS' or network_type == 'circuit_MPO':
            
            # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
            unitary = self.get_tensor(params)
            
            # if network_type is set to random-holoMPS:
            if network_type == 'random_state': 
                
                # defining tensor dimensions
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) # change to MPS-based structure
                d = tensor[:,0,0].size # physical leg dimension (for random state)
                chi = tensor[0,:,0].size # bond leg dimension (for random state)
                
                # change the order of indices to (p_out, b_in, b_out)
                # random selection of each physical site (and bulk tensors of structure)
                tensor_list = [np.swapaxes(unitary[:,:,j,:],1,2) for j in range(d)]  
                tensor_list1 = [random.choice(tensor_list) for j in range(L)]
            
            # if network_type is set to holoMPS:
            elif network_type == 'circuit_MPS':
                
                # defining tensor dimensions
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) # change to MPS-based structure
                d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
                chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
            
                # bulk tensors of holoMPS structure
                tensors = L*[unitary]  
                # change the order of indices to (p_out, b_in, b_out) 
                # (with p_in = 0 to go from unitary to isometry)
                tensor_list1 = [np.swapaxes(unitary[:,:,0,:],1,2) for unitary in tensors]
            
            # if network_type is set to circuit_MPO 
            # this option assumes original, circuit-based MPO structures (e.g. holoMPO)
            elif network_type == 'circuit_MPO':
                
                # defining tensor dimensions (consistent with rank-4 structures)
                # index ordering consistent with holographic-based MPO structures
                d = unitary[:,0,0,0].size # physical leg dimension (for MPO)
                chi = unitary[0,:,0,0].size # bond leg dimension (for MPO)
                tensor_list1 = L*[unitary]
            
            # testing boundary conditions 
            bdry = []
            if network_type == 'random_state' or network_type == 'circuit_MPS': # specific to holoMPS-based structures
                # if both boundary vectors are specified 
                for j in range(2):
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                        bdry.append(bdry_vecs[j])
                        
                # if boundary vectors are not specified for holoMPS-based structures:     
                # checking the left boundary vector
                if np.array(bdry_vecs[0] == None).all():
                    # if left boundary vector not specified, set to (1,0,0,0...)
                    bdry += [np.zeros(chi)]
                    bdry[0][0] = 1
                else:
                    if bdry_vecs[0].size != chi:
                        raise ValueError('left boundary vector different size than bulk tensors')
                    bdry += [bdry_vecs[0]]
                
                # checking the right boundary vector (special to holoMPS-based structures)
                if np.array(bdry_vecs[1] == None).all():
                    bdry += [None]
                else:
                    if bdry_vecs[1].size != chi:
                        raise ValueError('right boundary vector different size than bulk tensors')
                    bdry += [bdry_vecs[1]]
                    
            elif network_type == 'circuit_MPO': # specific to holoMPO-based structures       
                for j in range(2):
                    # if both boundary vectors are specified 
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                        bdry.append(bdry_vecs[j])
                    
                    elif np.array(bdry_vecs[j] == None).all():
                        # if boundary vectors not specified, set to (1,0,0,0...)
                        bdry += [np.zeros(chi)]
                        bdry[j][0] = 1
                    else:
                        if bdry_vecs[j].size != chi:
                            raise ValueError('boundary vectors different size than bulk tensors')
                        bdry += [bdry_vecs[j]]         
            
            M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
                      
        # if network_type is set to MPO: 
        # this option assumes genuine MPO_based structures (e.g. Hamiltonian MPO)  
        elif network_type == 'MPO':           
            # only the bulk tensors of the main chain must be included (w/out params)
            tensor_list1 = L*[self]
            
            # testing boundary conditions
            bdry = []
            for j in range(2):
                # if both boundary vectors are specified 
                if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_MPO:
                    bdry.append(bdry_vecs[j])
                    
                elif np.array(bdry_vecs[j] == None).all():
                    # if boundary vectors not specified, set to (1,0,0,0...)
                    bdry += [np.zeros(chi_MPO)]
                    bdry[j][0] = 1
                else:
                    if bdry_vecs[j].size != chi_MPO:
                        raise ValueError('boundary vectors different size than bulk tensors')
                    bdry += [bdry_vecs[j]]
                
            M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
        
        return M
    
    def prob_list(self, state_type, circuit, params):
        """  
        Returns the list of probability weights of each physical state for random-holographic 
        matrix prodcut states or density matrix.
        --------------
        Inputs:
          --the input accepts thermal_state_class-based random_state (random-holoMPS) or 
            should be set to None by default for density matrix evaluations--
          state_type: str
             One of "random_state" or "density_matrix" options.
          circuit: holographic-based circuit structure.
          params: numpy.ndarray
             Parameters of circuit structure. This could also be any randomly generated 
             numpy.ndarray structure consistent with bulk tensor physical leg dimension.
        """
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = circuit.get_tensor(params)
        
        # for random holo-MPS-based structures:
        if state_type == 'random_state': 
            
            # tensor dimensions (consistent with rank-3 structure)
            d = self[1][0][:,0,0].size # physical leg dimension (for random state) 
            chi = self[1][0][0,:,0].size # bond leg dimension (for random state)
            count = [0]*d
            
            # change the order of indices to (p_out, b_in, b_out)
            tensor_list = [np.swapaxes(unitary[:,:,j,:],1,2) for j in range(d)]  
            # counting the number of individual physical sites in tensor_list 
            # (by comparing with random state's bulk tensors)
            for tensor in self[1]:
                for j in range(d):
                    if (tensor == tensor_list[j]).all():
                        count[j] += 1
                prob_list = [c/L for c in count] # list of probability weights
        
        # for density-matrix-based structures:
        elif state_type == 'density_matrix':
            
            # tensor dimensions (consistent with rank-4 structure)
            # index ordering consistent with holographic-based MPO structures
            d = unitary[:,0,0,0].size # physical leg dimension
            chi = unitary[0,:,0,0].size # bond leg dimension
            p1_list = params[:d]
            prob_list = [p/sum(p1_list) for p in p1_list] # list of probability weights
       
        return prob_list

    def density_matrix(self, params, L, prob_list=None, bdry_vecs=[None,None]):      
        """
        Returns Density Matrix Product Operator (DMPO) of a tensor network.
        --------------
        Inputs:
          --the input accepts holographic-based circuit structures--
          params: numpy.ndarray
             Parameters of circuit structure.
          L: int
             length (number) of repetitions of unit cell in the main network chain.
          prob_list: list 
             List of probability weights of each physical state (the length of prob_list 
             should match the physical leg dimension). If set to None, it would call
             thermal_based prob_list fuction to compute probability weights for density
             matrix.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default which gives left and right boundary vectors = |0>).
        """
        
        # checking whether the prob. weights list is given (and its size)
        if prob_list != None:
            
            # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
            unitary = c.get_tensor(params)  
        
            # tensor dimensions (consistent with rank-4 structure)
            # index ordering consistent with holographic-based MPO structures
            d = unitary[:,0,0,0].size # physical leg dimension
            chi = unitary[0,:,0,0].size # bond leg dimension
            if len(prob_list) != d: # checking the size of probability weights list
                raise ValueError('length of probability list should match the physical dimension') 
        
        # if prob_list set to None (by default):
        elif prob_list == None:
            prob_list = thermal_state.prob_list(None,'density_matrix',self,params)
            
        # constructing state and probability weights network chain (as MPO)
        state = thermal_state.network_from_cells(c,'circuit_MPO',L,None,params,[None,None])
        p_matrix = np.diag(prob_list)
        p_state = thermal_state.network_from_cells(p_matrix,'MPO',L,d,None,[None,None])
     
        
        # contractions of density matrix: 
        contractions = []
        for j in range(L):
            # contracting the prob. weights chain with state
            t = np.tensordot(p_state[1][j],state[1][j],axes=[0,2]) # result:  p'_in, p_out, b_out, b_in
            t1 = np.swapaxes(t,0,2) # changing axis ordering to:  b_out, p_out, p'_in, b_in
            t2 = np.swapaxes(t1,0,1) # changing axis ordering to:  p_out, b_out, p'_in, b_in
            #contracting results with dual state
            t3 = np.tensordot(state[1][j].conj(),t2,axes=[2,2]) # contracting p_in and p'_in  
            contractions.append(t3)
           
        # boundary contractions
        # state boundary contractions
        bvecl_s = np.kron(p_state[0][0],state[0][0]) 
        bvecr_s = np.kron(p_state[2][0],state[2][0])
        # boundary contractions with dual state 
        bvecl_tot = np.kron(state[0][0].conj(),bvecl_s) 
        bvecr_tot = np.kron(state[2][0].conj(),bvecr_s)
        
        density_matrix = [[bvecl_tot],contractions,[bvecr_tot]]
        
        return density_matrix

    def transfer_matrix(self, state_type, L, chi_MPO=None, MPO=None):
        """
        Returns transfer-matrix-like structures of a given matrix product state 
        (which might also include a matrix product operator in-between the states)
        or returns a transfer-matrix-like struture of a density matrix with an MPO 
        (e.g. Hamiltonian MPO).
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and DMPO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          L: int
             length (number) of repetitions of unit cell in the main network chain.
          chi_MPO: int
             Bond leg dimension for MPO-based structures. 
          MPO: thermal_state_class-based MPO structure.  
             Set to None for pure wave function simulations.
        Note:
          -If MPO is not inserted for holoMPS states, the function computes the transfer
           matrix for the state wave fucntion.
        """
        
        # for holoMPS and random holoMPS-based structures: 
        if state_type == 'random_state' or state_type == 'circuit_MPS':

            # tensor dimensions (consistent with rank-3 structures)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension 
            chi = tensor[0,:,0].size # bond leg dimension 
        
            # transfer matrix for the wave function: 
            # (without MPO inserted)
            if MPO == None: 
                t_mat_site = np.zeros([chi**2,chi**2],dtype=complex) # transfer matrix at each site
                t_mat = np.eye(chi**2,chi**2,dtype=complex) # total transfer matrix 
        
                # site contractions for state and its dual
                for j in range(L):
                    t_tensor =  np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix
                    t_mat_site = np.reshape(np.swapaxes(t_tensor,1,2),[chi**2,chi**2]) 
                    t_mat = t_mat_site @ t_mat # t-mat accumulations                 
        
            # transfer matrix w/ unit cell of MPO inserted
            else:
                t_mat_site = np.zeros([chi**2 * chi_MPO,chi**2 * chi_MPO]) # transfer matrix at each site
                t_mat = np.eye(chi**2 * chi_MPO,chi**2 * chi_MPO,dtype=complex) # total transfer matrix
            
                # site contractions for state/MPO/dual-state
                for j in range(L):
                    # contractions with state
                    chi_s = chi_MPO * chi
                    MPO_on_state = np.tensordot(MPO[1][j],self[1][j],axes=[2,0])
                    MPO_on_state_reshape = np.reshape(np.swapaxes(MPO_on_state,2,3),(d,chi_s,chi_s))

                    # constractions with dual state
                    chi_tot = chi_s * chi # total bond dimension
                    t_mat_site = np.reshape(np.swapaxes(np.tensordot(self[1][j].conj(),MPO_on_state_reshape,axes=[0,0]),1,2),[chi_tot,chi_tot])
                    t_mat = t_mat_site @ t_mat # t-mat accumulations
        
        # transfer-matrix-like structure for density matrix
        elif state_type == 'density_matrix':
            
            # constructing the transfer-matrix-like structure
            chi_tot = int(np.sqrt((np.tensordot(MPO[1][0],self[1][0],axes=[2,0])).size)) # total bond dimension
            t_mat_site = np.zeros([chi_tot,chi_tot],dtype=complex) # transfer matrix at each site
            t_mat = np.eye(chi_tot,chi_tot,dtype=complex) # total transfer matrix 
            
            for j in range(L):
                # MPO and density matrix constractions
                TN1 = np.tensordot(self[1][j],MPO[1][j],axes=[0,2])
                t_mat_site = np.reshape(TN1,[chi_tot,chi_tot])         
                t_mat = t_mat_site @ t_mat # t-mat accumulations
            
        return t_mat

    def expectation_value(self, state_type, L, chi_MPO=None, MPO=None):
        """
         Returns the result of <MPS|MPO|MPS> or <MPO|DMPO> contractions.
         MPS: holographic-matrix-product-state-based structures.
         MPO: matrix product operator.
         DMPO: density matrix operator.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and DMPO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          L: int
             length (number) of repetitions of unit cell in the main network chain.
          chi_MPO: int
             Bond leg dimension for MPO-based structures.
          MPO: thermal_state_class-based MPO structure.
             Set to None for pure wave function simulations.
        Note:
          -Left boundary condition is set by the given holoMPS boundary vectors, and the right 
           condition is averaged over (as consistent with holographic-based simulations).
          -If MPO is not inserted (for MPS structures), the function computes the expectation value 
           for the state wave fucntion (<MPS|MPS>).
        """
        
        t_mat = thermal_state.transfer_matrix(self,state_type,L,chi_MPO,MPO) # transfer-matrix-like structure
        
        # for holoMPS and random holoMPS-based structures:
        if state_type == 'random_state' or state_type == 'circuit_MPS':
           
            # tensor dimensions (consistent with rank-3 structure)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
            chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
    
            # without MPO inserted
            if MPO == None:
                bvecl = np.kron(self[0][0].conj(),self[0][0]) # left boundary contraction
                # right boundary constraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vectors if right boundary condition is not specified 
                    t_mat_on_rvec = np.reshape(t_mat @ bvecl,[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec, t_mat_on_rvec)
                else:
                    bvecr = np.kron(self[2][0].conj(),self[2][0])
                    expect_val = bvecr.conj().T @ t_mat @ bvecl
        
           # with MPO inserted
            else:
                bvecl = np.kron(self[0][0].conj(),np.kron(MPO[0][0],self[0][0])) # left boundary contraction
                # right boundary constraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vectors if right boundary condition is not specified
                    # employ the specified right boundary vector of the MPO.                 
                    t_vleft = np.reshape((t_mat @ bvecl),[chi,chi_MPO,chi]) # t_mat on left vector
                    MPO_rvec_contracted = np.reshape(np.tensordot(MPO[2][0],t_vleft,axes=[0,1]),[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,MPO_rvec_contracted)
                else:
                    bvecr = np.kron(self[2][0].conj(),np.kron(MPO[2][0],self[2][0]))
                    expect_val = bvecr.conj().T @ t_mat @ bvecl
        
        # for density-matrix-based structures:
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix':
            
            # boundary vector contractions
            bvecl = np.kron(self[0][0],MPO[0][0])
            bvecr = np.kron(self[2][0],MPO[2][0])
            expect_val = bvecr.conj().T @ t_mat @ bvecl
            
        return (expect_val).real

    def free_energy(self, params, state_type, L, H_mat, T, chi_H, prob_list=None, bdry_vecs1=[None,None], bdry_vecs2=[None,None]):
        """
         Returns the Helmholtz free energy of a density matrix or random holographic 
         matrix product state.
        --------------
        Inputs:
        --the input accepts holographic-based circuit structures--
        state_type: str
           One of "density_matrix" or "random_state" options.
        L: int 
           Length (number) of repetitions of unit cell in the main network chain.
        params: numpy.ndarray
             Parameters of circuit structure.
        H_mat: numpy.ndarray 
           The unit cell of the Hamiltonian MPO of model.  
        T: float
           Tempreture.
        chi_H: int
             Bond leg dimension for Hamiltonian MPO structure.
        prob_list (for "density_matrix" option): list 
             List of probability weights of each physical state (the length of prob_list 
             should match the physical leg dimension). If set to None, it would call
             thermal_based prob_list fuction to compute probability weights for density
             matrix.
        bdry_vecs1 and bdry_vecs2: list
            List of left (first element) and right (second element) boundary vectors for 
            state and Hamiltonian networks, respectively (set to [None,None] by default).         
        """     
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = self.get_tensor(params)
        
        # for density-matrix-based structures:
        if state_type == 'density_matrix':
            
            # if prob_list set to None (by default): 
            if prob_list == None:
                prob_list = thermal_state.prob_list(None,'density_matrix',self,params)
            
            # constructing state and Hamiltonian MPO          
            density_mat = thermal_state.density_matrix(self,params,L,prob_list,bdry_vecs1)
            Hamiltonian = thermal_state.network_from_cells(H_mat,'MPO',L,chi_H,None,bdry_vecs2) 
            
            # entropy calculations 
            S = 0
            S_list = []
            # checking purity of state
            for p in prob_list:
                if p != 0:
                    S_list.append(-p*np.log(p))
            for j in range(len(S_list)):
                S = S + S_list[j]
        
            E = thermal_state.expectation_value(density_mat,'density_matrix',L,chi_H,Hamiltonian) # energy of system    
            F = E - T*S # Helmholtz free energy
        
        # for random-holoMPS structures:
        elif state_type == 'random_state':
            
            # constructing random state, Hamiltonian, and prob_list (for random state)
            random_state = thermal_state.network_from_cells(self,'random_state',L,None,params,bdry_vecs1)
            Hamiltonian = thermal_state.network_from_cells(H_mat,'MPO',L,chi_H,None,bdry_vecs2)
            prob_list = thermal_state.prob_list(random_state,'random_state',self,params)
            
            # entropy calculations 
            S = 0
            S_list = []
            # checking purity of state
            for p in prob_list:
                if p != 0:
                    S_list.append(-p*np.log(p))
            for j in range(len(S_list)):
                S = S + S_list[j]
                
            E = thermal_state.expectation_value(random_state,'random_state',L,chi_H,Hamiltonian) # energy of system    
            F = E - T*S # Helmholtz free energy      
        return F    
