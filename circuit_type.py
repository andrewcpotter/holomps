from circuit_qubit import Circuit

def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

# c is a circuit object
# site_list is a length 4 list, it includes the qubit indices which the gates in this ansatz act on
def bond_4_optimize_type(c, site_list):
    s0 = site_list[0]
    s1 = site_list[1]
    s2 = site_list[2]
    s3 = site_list[3]
    # rotation Z gate on one physical qubit
    #c.add_gate("rotation", qids = [s0], n_params = 1, fn = qub_z)
    
    # entangle p_up & b_up, p_down & b_down
    c.add_gate("XX_YY", qids=[s0, s2], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids=[s0, s2], n_params = 1, fn = qub_two)
    c.add_gate("XX_YY", qids=[s1, s3], n_params = 1, fn = qub_two)    
    c.add_gate("ZZ", qids=[s1, s3], n_params = 1, fn = qub_two)
    
    # entangle p_up & p_down, b_up & b_down
    c.add_gate("XX_YY", qids=[s0, s1], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [s0, s1], n_params = 1, fn = qub_two)
    c.add_gate("XX_YY", qids=[s2, s3], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids = [s2, s3], n_params = 1, fn = qub_two)
	
    # entangle p_up & b_up, p_down & b_down again
    c.add_gate("XX_YY", qids=[s0, s2], n_params = 1, fn = qub_two)
    c.add_gate("ZZ", qids=[s0, s2], n_params = 1, fn = qub_two)
    c.add_gate("XX_YY", qids=[s1, s3], n_params = 1, fn = qub_two)    
    c.add_gate("ZZ", qids=[s1, s3], n_params = 1, fn = qub_two)
    
    #c.add_gate("rotation", qids = [s0], n_params = 1, fn = qub_z)
    return c    

# for now assume c includes 6 qubits (2 physical qubits and 4 bond qubits) for bond dim = 16 Fermi-Hubbard
# need to generalize later
def star_circuit(c):
    c = bond_4_optimize_type(c, [0,1,2,3])
    c = bond_4_optimize_type(c, [0,1,4,5])
    return c

# for now assume c includes 6 qubits (2 physical qubits and 4 bond qubits) for bond dim = 16 Fermi-Hubbard
# need to generalize later
def fully_connected_circuit(c):
    c = bond_4_optimize_type(c, [0,1,2,3])
    c = bond_4_optimize_type(c, [0,1,4,5])
    c = bond_4_optimize_type(c, [2,3,4,5])
    return c