# note that in this circuit type file, there are some redunancies in the code, which can be simplied by 
# writing circuit in heriachy structure, this will be updated soon.
from circuit_qubit import Circuit

def Rot_x(params): return (np.pi/2, 0, params[0])
def Rot_y(params): return (np.pi/2, np.pi/2, params[0])
def Rot_z(params): return (0, 0, params[0])
def two_qub_gate(params): return (params[0])

# construct the fixed ansatz for 4-qubit circuit
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

# for now assume c includes 6 qubits (2 physical qubits and 4 bond qubits) for bond dim = 16 Fermi-Hubbard
# need to generalize later
def star_circuit(c):
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

# for now assume c includes 6 qubits (2 physical qubits and 4 bond qubits) for bond dim = 16 Fermi-Hubbard
# need to generalize later
def fully_connected_circuit(c):
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
    
    # second layer, similiar to first layer, but entangle four bond qubits:
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
    
    # third layer, basically repeat first layer, instead let two physical qubits entangle with 
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