

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
          --the input assumes either circuit or rank-4 numpy.ndarray tensor--       
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
        if network_type != "random_state" and network_type != "circuit_MPS" and network_type != "circuit_MPO" and network_type != "MPO": 
                raise ValueError('only one of "random_state", "circuit_MPS", "circuit_MPO", "MPO" options')
                
        # for circuit-based structures:
        # both circuit and params must be included
        if network_type == 'random_state' or network_type == 'circuit_MPS' or network_type == 'circuit_MPO':
            
            # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
            unitary = self.get_tensor(params[:self.n_params])
            
            # if network_type is set to random-holoMPS:
            if network_type == 'random_state': 
            
                # defining tensor dimensions
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) # change to MPS-based structure
                d = tensor[:,0,0].size # physical leg dimension (for random state)
                chi = tensor[0,:,0].size # bond leg dimension (for random state)
                         
                prob_list = thermal_state.prob_list(self,params) # list of variational probability weights
                # converting probability weights to numbers for each physical state in random MPS
                num_list = [int(round(p*L)) for p in prob_list[1:len(prob_list)]]
                num_list.append(L-sum(num_list))
                
                # change the order of indices to (p_out, b_in, b_out)
                tensor_list = [[np.swapaxes(unitary[:,:,j,:],1,2)]*n for j,n in zip(range(d),num_list)]
                tensor_list1 = []
                for element in tensor_list:
                    for j in range(len(element)):
                        tensor_list1.append(element[j])
                
                # random selection of each physical site by random shuffling in final state structure
                random.shuffle(tensor_list1) 

            # if network_type is set to holoMPS:
            elif network_type == 'circuit_MPS':
            
                # defining tensor dimensions
                # change the order of indices to (p_out, b_in, b_out) 
                # (with p_in = 0 to go from unitary to isometry)
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) 
                d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
                chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
                
                # bulk tensors of holoMPS structure
                tensor_list1 = [tensor]*L  

            # if network_type is set to circuit_MPO 
            # this option assumes original, circuit-based MPO structures (e.g. holoMPO)
            elif network_type == 'circuit_MPO':
                
                # defining tensor dimensions (consistent with rank-4 structures)
                # index ordering consistent with holographic-based MPO structures
                d = unitary[:,0,0,0].size # physical leg dimension (for MPO)
                chi = unitary[0,:,0,0].size # bond leg dimension (for MPO)
                tensor_list1 = [unitary]*L
            
            # testing boundary conditions 
            bdry = []
            if network_type == 'random_state' or network_type == 'circuit_MPS': # specific to holoMPS-based structures 
                
                # if boundary vectors are not specified for holoMPS-based structures:     
                # checking left boundary vector
                # if left boundary vector not specified, set to (1,0,0,0...)
                if np.array(bdry_vecs[0] == None).all():
                    bdry += [np.zeros(chi)]
                    bdry[0][0] = 1
                else:
                    if bdry_vecs[0].size != chi:
                        raise ValueError('left boundary vector different size than bulk tensors')
                    bdry += [bdry_vecs[0]]
                
                # checking right boundary vector (special to holoMPS-based structures)
                if np.array(bdry_vecs[1] == None).all():
                    bdry += [None]
                else:
                    if bdry_vecs[1].size != chi:
                        raise ValueError('right boundary vector different size than bulk tensors')
                    bdry += [bdry_vecs[1]]
                    
                # if both boundary vectors are specified 
                for j in range(2):
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                        bdry.append(bdry_vecs[j]) 
                        
            elif network_type == 'circuit_MPO': # specific to holoMPO-based structures       
                for j in range(2):
                    # if both boundary vectors are specified 
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                        bdry.append(bdry_vecs[j])
                    
                    # if boundary vectors not specified, set to (1,0,0,0...)
                    elif np.array(bdry_vecs[j] == None).all():
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
            # only bulk tensors of the main chain must be included (w/out params)
            tensor_list1 = [self]*L
            
            # testing boundary conditions
            bdry = []
            for j in range(2):
                # if both boundary vectors are specified 
                if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_MPO:
                    bdry.append(bdry_vecs[j])
                
                # if boundary vectors not specified, set to (1,0,0,0...)
                elif np.array(bdry_vecs[j] == None).all():
                    bdry += [np.zeros(chi_MPO)]
                    bdry[j][0] = 1
                else:
                    if bdry_vecs[j].size != chi_MPO:
                        raise ValueError('boundary vectors different size than bulk tensors')
                    bdry += [bdry_vecs[j]]
                
            M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
        
        return M
    
    def prob_list(self, params):
        """  
        Returns the list of variational probability weights of each physical state for 
        random-holographic matrix product state or density matrix.
        --------------
        Inputs:
          --the input accepts holographic-based circuit structures--
          params: numpy.ndarray
             Parameters of circuit structure. This could also be any randomly generated 
             numpy.ndarray structure consistent with bulk tensor physical leg dimension.
        """
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = self.get_tensor(params[:self.n_params])
        list1 = params[self.n_params:]
        prob_list = [p/sum(list1) for p in list1]
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
             Length (number) of repetitions of unit cell in the main network chain.
          prob_list: list 
             List of probability weights of each physical state (the length of prob_list 
             should match the physical leg dimension). If set to None, it would call
             thermal_based prob_list fuction to compute probability weights for density
             matrix.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default which gives left and right boundary vectors = |0>).
        """
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = self.get_tensor(params[:self.n_params])  
        
        # tensor dimensions (consistent with rank-4 structure)
        # index ordering consistent with holographic-based MPO structures
        d = unitary[:,0,0,0].size # physical leg dimension
        chi = unitary[0,:,0,0].size # bond leg dimension
        
        if prob_list != None: # checking the size of a given probability weight list 
            if len(prob_list) != d: 
                raise ValueError('length of probability list should match the physical dimension') 
                
        # constructing state and probability weights network chain (as MPO)
        state = thermal_state.network_from_cells(self,'circuit_MPO',L,None,params,bdry_vecs)
        # diagonal matrices of probability weights chain
        if prob_list != None:
            p_matrix = np.diag(prob_list)
        else:
            p_matrix = np.diag(thermal_state.prob_list(self,params))      
        p_state = thermal_state.network_from_cells(p_matrix,'MPO',L,d,None,[None,None])
        
        # contractions of density matrix: 
        contractions = []
        for j in range(L):
            # contracting the probability weights chain with state
            # changing axis ordering to: b_out, p_out, p'_in, b_in
            W1 = np.swapaxes(np.tensordot(p_state[1][j],state[1][j],axes=[0,2]),0,2) 
            W2 = np.swapaxes(W1,0,1) # changing axis ordering to:  p_out, b_out, p'_in, b_in
            #contracting results with dual state
            W3 = np.tensordot(state[1][j].conj(),W2,axes=[2,2]) # contracting p_in and p'_in  
            contractions.append(W3)
           
        # boundary contractions
        # state boundary contractions
        bvecl_s = np.kron(p_state[0][0],state[0][0]) 
        bvecr_s = np.kron(p_state[2][0],state[2][0])
        # boundary contractions with dual state 
        bvecl_tot = np.kron(state[0][0].conj(),bvecl_s) 
        bvecr_tot = np.kron(state[2][0].conj(),bvecr_s)
        
        density_matrix = [[bvecl_tot],contractions,[bvecr_tot]]
        
        return density_matrix

    def transfer_matrix(self, state_type, chi_MPO=None, MPO=None):
        """
        Returns transfer-matrix-like structures of a given matrix product state 
        (which might also include a matrix product operator in-between the states)
        or returns transfer-matrix-like strutures of a density matrix with an MPO 
        (e.g. Hamiltonian MPO).
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and DMPO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          chi_MPO: int
             Bond leg dimension for MPO-based structure. 
          MPO: thermal_state_class-based MPO structure.  
             Set to None for pure wave function simulations for MPS states.
        Note:
          -If MPO is not inserted for holoMPS states, the function computes the transfer
           matrices for the state wave fucntion at each site.
          -Length of MPO structure might be different than length of state.
          -The output would be returned as a list of transfer matrices computed for each 
           unit cell at each site.
        """
        if state_type != "random_state" and state_type != "circuit_MPS" and state_type != "density_matrix": 
                raise ValueError('only one of "random_state", "circuit_MPS", or "density_matrix" options')
                
        t_mat_list = []
        # for holoMPS and random holoMPS-based structures: 
        if state_type == 'random_state' or state_type == 'circuit_MPS':

            # tensor dimensions (consistent with rank-3 structures)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension 
            chi = tensor[0,:,0].size # bond leg dimension 
            L = len(self[1]) # length of repetitions of unit cell in main network chain (for holoMPS state).
            
            # transfer matrix for the wave function: 
            # (w/out MPO inserted)
            if MPO == None: 
                # site contractions for state and its dual
                for j in range(L):
                    # contraction state/dual state
                    t_tensor =  np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix
                    t_mat_site = np.reshape(np.swapaxes(t_tensor,1,2),[chi**2,chi**2]) # transfer matrix at each site
                    t_mat_list.append(t_mat_site)                 
     
            # transfer matrix w/ MPO inserted
            else: 
                L_MPO = len(MPO[1]) # length of repetitions of unit cell for squeezed MPO structure.
                # site contractions for state/MPO/dual-state
                for j in range(L_MPO):
    
                    # contractions with state
                    chi_s = chi_MPO * chi
                    MPO_on_state = np.tensordot(MPO[1][j],self[1][j],axes=[2,0])
                    MPO_on_state_reshape = np.reshape(np.swapaxes(MPO_on_state,2,3),[d,chi_s,chi_s])

                    # contractions with dual state
                    chi_tot = chi_s * chi # total bond dimension
                    # transfer matrix at each site
                    t_mat_site = np.reshape(np.swapaxes(np.tensordot(self[1][j].conj(),MPO_on_state_reshape,axes=[0,0]),1,2),[chi_tot,chi_tot])
                    t_mat_list.append(t_mat_site)
                
                # contraction of rest of state and its dual if L_MPO different than L
                for j in range(L-L_MPO):
                    # contraction state/dual state
                    t_tensor =  np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix
                    t_mat_site = np.reshape(np.swapaxes(t_tensor,1,2),[chi**2,chi**2]) # transfer matrix at each site
                    t_mat_list.append(t_mat_site)     
        
        # transfer-matrix-like structure for density matrix
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix':
            
            L = len(self[1]) # length of repetitions of unit cell in main network chain (for density matrix).
            L_MPO = len(MPO[1]) # length of repetitions of unit cell for inserted MPO structure.
            chi_tot = int(np.sqrt((np.tensordot(self[1][0],MPO[1][0],axes=[0,2])).size)) # total bond dimension
            
            for j in range(L):
                # MPO and density matrix constractions
                t_tensor = np.tensordot(self[1][j],MPO[1][j],axes=[0,2])
                t_mat_site = np.reshape(t_tensor,[chi_tot,chi_tot]) # transfer-matrix-like structure at each site       
                t_mat_list.append(t_mat_site)
            
        return t_mat_list

    def expectation_value(self, state_type, chi_MPO=None, MPO=None):
        """
         Returns the result of <MPS|MPS>, <MPS|MPO|MPS>, or <MPO|DMPO> contractions.
         MPS: holographic-matrix-product-state-based structures.
         MPO: matrix product operator.
         DMPO: density matrix product operator.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and DMPO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
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
        if state_type != "random_state" and state_type != "circuit_MPS" and state_type != "density_matrix":
            raise ValueError('only one of "random_state", "circuit_MPS", or "density_matrix" options')
                
        t_mat = thermal_state.transfer_matrix(self,state_type,chi_MPO,MPO) # transfer-matrix-like structures
        # accumulation of transfer matrices defined at each site
        t_mat0 = t_mat[0]
        for j in range(1,len(t_mat)):
            t_mat0 = t_mat[j] @ t_mat0
        
        # for holoMPS and random holoMPS-based structures:
        if state_type == 'random_state' or state_type == 'circuit_MPS':
           
            # tensor dimensions (consistent with rank-3 structure)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
            chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
    
            # w/out MPO inserted
            if MPO == None:
                bvecl = np.kron(self[0][0].conj(),self[0][0]) # left boundary contraction
                # right boundary contraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vector if right boundary condition is not specified 
                    t_mat_on_rvec = np.reshape(t_mat0 @ bvecl,[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,t_mat_on_rvec)
                else:
                    bvecr = np.kron(self[2][0].conj(),self[2][0])
                    expect_val = bvecr.conj().T @ t_mat0 @ bvecl
        
           # w/ MPO inserted
            else:
                bvecl = np.kron(self[0][0].conj(),np.kron(MPO[0][0],self[0][0])) # left boundary contraction
                # right boundary constraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vectors if right boundary condition is not specified
                    # employ the specified right boundary vector of MPO.                 
                    t_vleft = np.reshape((t_mat0 @ bvecl),[chi,chi_MPO,chi]) # t_mat on left vector
                    MPO_rvec_contracted = np.reshape(np.tensordot(MPO[2][0],t_vleft,axes=[0,1]),[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,MPO_rvec_contracted)
                else:
                    bvecr = np.kron(self[2][0].conj(),np.kron(MPO[2][0],self[2][0]))
                    expect_val = bvecr.conj().T @ t_mat0 @ bvecl
        
        # for density-matrix-based structures:
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix':
            
            # boundary vector contractions
            bvecl = np.kron(self[0][0],MPO[0][0])
            bvecr = np.kron(self[2][0],MPO[2][0])
            expect_val = bvecr.conj().T @ t_mat0 @ bvecl
            
        return (expect_val).real
    
    def entropy(prob_list):
        """
        Returns the von Neumann entropy for a given list probability weights.
        --------------
        --the input assumes thermal_state_class-based prob_list--
        """
        S = 0
        S_list = []
        # checking purity of state
        for p in prob_list:
            if p != 0:
                S_list.append(-p*np.log(p))
        for j in range(len(S_list)):
            S = S + S_list[j]
        return S
    
    def free_energy(self, params, state_type, L, H_mat, T, chi_H, prob_list=None, bdry_vecs1=[None,None], bdry_vecs2=[None,None]):
        """
         Returns the Helmholtz free energy of a thermal density matrix structure 
         or thermal random holographic matrix product state.
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
        if state_type != "random_state" and state_type != "density_matrix":
            raise ValueError('only one of "random_state" or "density_matrix" options')
                
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = self.get_tensor(params[:self.n_params])
        
        # for density-matrix-based structures:
        if state_type == 'density_matrix':
            
            # checking whether the probability weight list is given  
            if prob_list != None:
                S = thermal_state.entropy(prob_list) # entropy 
                density_mat = thermal_state.density_matrix(self,params,L,prob_list,bdry_vecs1) # density matrix
            else:
                S = thermal_state.entropy(thermal_state.prob_list(self,params)) # entropy
                density_mat = thermal_state.density_matrix(self,params,L,None,bdry_vecs1) 
                
            Hamiltonian = thermal_state.network_from_cells(H_mat,'MPO',L,chi_H,None,bdry_vecs2) # Hamiltonian MPO      
            E = thermal_state.expectation_value(density_mat,'density_matrix',chi_H,Hamiltonian) # energy of system    
            F = E - T*S # Helmholtz free energy
            
        # for random-holoMPS-based structures:
        elif state_type == 'random_state':
            
            random_state = thermal_state.network_from_cells(self,'random_state',L,None,params,bdry_vecs1) # random_state MPS
            Hamiltonian = thermal_state.network_from_cells(H_mat,'MPO',L,chi_H,None,bdry_vecs2) # Hamiltonian MPO
            S = thermal_state.entropy(thermal_state.prob_list(self,params)) # entropy
            E = thermal_state.expectation_value(random_state,'random_state',chi_H,Hamiltonian) # energy of system    
            F = E - T*S # Helmholtz free energy   
        return F
