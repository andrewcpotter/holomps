'''
construct the circuit ansatz for fermi-hubbard model VQE in bond dimension 4 and 16, 
here for bond dimension = 4 we have 4 qubits in total, two physical qubits and two bond qubits, 
in the comments denote them by physical qubits: p_up, p_down; bond qubits: b_up, b_down
for bond dimension = 16 we have 6 qubits in total, two physical qubits and four bond qubits.
'''
from circuit_qubit import Circuit

# parameter functions for Z rotation gate and two qubit gates used here in general, including ZZ and XXYY gates
def Rot_z(params): return (0, 0, params[0])
def two_qub_gate(params): return (params[0])
'''

an ansatz for 4-qubit circuit, right now the best ansatz for bond dim 4 -- get the best VQE results
included two same layer of entanglement physical with bond qubits, with one layer of entangling 
physical(/bond) qubits with physical(/bond) qubits in between
'''
def bond_4_optimize_type(c):
    # rotation Z gate on one physical qubit
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    
    # entangle p_up & b_up, p_down & b_down
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    # entangle p_up & p_down, b_up & b_down
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = two_qub_gate)
	
    # entangle p_up & b_up, p_down & b_down again
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    c.assemble()
    return c

'''
a bond 4 circuit ansatz with simpler structure, proved to not work very well in VQE in previous tests
but takes fewer varying parameters, keep here for possible future needs
'''
def bond_4_easy(c):
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    c.assemble()
    return c        

'''
star circuit for bond dimension 16
include 6 qubits in total: 2 physical qubits and 4 bond qubits
include two layers, first layer repeat the best ansatz in bond dim 4
second layer is similar to the first layer, 
but for two physical qubits and the two added bond qubits
Notice that this structure is general, and called star circuit according 
to Drew's paper, we may need to generalize the code structure later
(after find a better way to assemble the circuit
'''
def star_circuit(c):
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = two_qub_gate)
	
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    # second layer, basically repeat first layer, instead let two physical qubits entangle with 
    # two other bond qubits
    c.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [4, 5], n_params = 1, fn = two_qub_gate)
	
    c.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = two_qub_gate)
        
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    c.assemble()
    return c

'''
fully connected circuit ansatz for bond dim 16
include three layers, add one layer of entangling between four bond qubits 
compared to star circuit
may need generalize the structure later 
'''
def fully_connected_circuit(c):
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    
    # first block
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = two_qub_gate)
	
    c.add_gate("XX_YY", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 2], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 3], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 3], n_params = 1, fn = two_qub_gate)
    
    # second block
    c.add_gate("XX_YY", qids=[2, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[2, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[3, 5], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[3, 5], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("XX_YY", qids=[2, 3], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [2, 3], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [4, 5], n_params = 1, fn = two_qub_gate)
	
    c.add_gate("XX_YY", qids=[2, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[2, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[3, 5], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[3, 5], n_params = 1, fn = two_qub_gate)
    
    # third block
    c.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = two_qub_gate)
    
    c.add_gate("XX_YY", qids=[0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [0, 1], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[4, 5], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids = [4, 5], n_params = 1, fn = two_qub_gate)
	
    c.add_gate("XX_YY", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("ZZ", qids=[0, 4], n_params = 1, fn = two_qub_gate)
    c.add_gate("XX_YY", qids=[1, 5], n_params = 1, fn = two_qub_gate)    
    c.add_gate("ZZ", qids=[1, 5], n_params = 1, fn = two_qub_gate)
        
    c.add_gate("rotation", qids = [0], n_params = 1, fn = Rot_z)
    c.assemble()
    return c