import time
import numpy as np
from circuit_qubit import Circuit
from tenpy.networks.mps import MPS
from tenpy.networks.site import FermionSite
# here consider spinless fermions 
from tenpy.models.fermions_spinless import FermionModel
from scipy.optimize import minimize
from hubbard_dmrg import Hubbard_spinless_fermion_dmrg

d = 2
chimax = 2
J_value = 2.0
mu_value = 1.0
V_value = 1.0

c = Circuit([("qubit", "p", d), ("qubit", "b", chimax)])

def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

# one qubit rotation
#c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_x)
#c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

#c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_x)
#c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_z)

# XX rotation
c.add_gate("XX", qids=[0, 1], n_params = 1, fn = qub_two)

# YY rotation 
c.add_gate("YY", qids=[0, 1], n_params = 1, fn = qub_two)    

# ZZ rotation
c.add_gate("ZZ", qids=[0, 1], n_params = 1, fn = qub_two)


# one qubit rotation
#c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_x)
#c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [0], n_params = 1, fn = qub_z)

#c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_x)
#c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_y)
c.add_gate("rotation", qids = [1], n_params = 1, fn = qub_z)

c.assemble()

def circuit_imps(params, circuit):
    site = FermionSite(conserve = None, filling = 0.5)
    # evaluate circuit, get rank-4 (p_out, b_out, p_in, b_in) unitary
    unitary = circuit.get_tensor(params)
    # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # (with p_in = 0 to go from unitary to isometry)
    B = [np.swapaxes(unitary[:,:,0,:],1,2)]
    psi = MPS.from_Bflat([site], B, bc='infinite', dtype=complex, form=None)    
    if psi.form is not None:
        try:
            psi.canonical_form()
            psi.convert_form(psi.form)
        except:
            print("psi form thing didn't work")
    return psi

def energy(params, circuit, Hamiltonian, psi):
    psi = circuit_imps(params, circuit)
    E = (Hamiltonian.expectation_value(psi)).real
    return E

rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi, size=c.n_params)
psi = circuit_imps(params, c)
print("norm of wave function = {0}".format(psi.norm))
print()

Fermion_spinless_Hubbard1 = FermionModel({'bc_MPS':'infinite', 'conserve':None, 'J':J_value, 'mu':mu_value, 'V':V_value})
Hamiltonian = Fermion_spinless_Hubbard1.calc_H_MPO()
t1 = time.time()
result = minimize(energy, x0=params, args=(c, Hamiltonian, psi), method='nelder-mead')
sweet_spot = result.x
t2 = time.time()

print("it took {}s to optimize".format(t2-t1))
print("sweet spot = {}".format(sweet_spot))
print("num function evaluations: {}".format(result['nfev']))
print("num iterations: {}".format(result['nit']))
print("termination msg: {}".format(result['message']))

holo_gs = circuit_imps(sweet_spot, c)
holo_E = energy(sweet_spot, c, Hamiltonian, holo_gs)
E0 = Hubbard_spinless_fermion_dmrg(J_value, mu_value, V_value)
print("holoMPS: E = {:.8}".format(holo_E))
print(" theory: E = {:.8}".format(E0))
print("  error: {:.7%}".format(np.abs(1 - holo_E/E0)))