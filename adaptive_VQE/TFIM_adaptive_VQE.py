import time
import numpy as np
from circuit_qubit_simplified import Circuit
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
import numpy as np
import scipy.sparse as ss

#from tenpy.models.tf_ising import TFIModel
#from qubit_circuit_with_JW1 import *

from scipy.optimize import minimize
from scipy.optimize import dual_annealing

d = 2
chimax = 2
J = 1
g = 1

c = Circuit([("qubit", "p", d), ("qubit", "b", chimax)])

def qub_x(params): return (np.pi/2, 0, params[0])
def qub_y(params): return (np.pi/2, np.pi/2, params[0])
def qub_z(params): return (0, 0, params[0])
def qub_two(params): return (params[0])

def circuit_imps(params, circuit):
    site = SpinHalfSite(conserve=None)
    # evaluate circuit, get rank-4 (p_out, b_out, p_in, b_in) unitary
    unitary = circuit.get_tensor(params)
    # change the order of indices to [p, vL, vR] = [p_out, b_in, b_out] 
    # (with p_in = 0 to go from unitary to isometry)
    B = np.swapaxes(unitary[:,:,0,:],1,2)
    psi = MPS.from_Bflat([site], [B], bc='infinite', dtype=complex, form=None)    
    if psi.form is not None:
        try:
            psi.canonical_form()
            psi.convert_form(psi.form)
        except:
            print("psi form thing didn't work")
    return psi

def ising_impo(J, g):
    site = SpinHalfSite(None)
    Id, Sp, Sm, Sz = site.Id, site.Sp, site.Sm, 2*site.Sz
    Sx = Sp + Sm
    W = [[Id,Sx,g*Sz], [None,None,-J*Sx], [None,None,Id]]
    H = MPO.from_grids([site], [W], bc='infinite', IdL=0, IdR=-1)
    return H


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

def energy(params, circuit, Hamiltonian):
    psi = circuit_imps(params, circuit)
    E = (Hamiltonian.expectation_value(psi)).real
    return E

rng = np.random.default_rng() 
params = rng.uniform(high=2*np.pi, size=c.n_params)
psi = circuit_imps(params, c)
print(psi)
print("norm of wave function = {0}".format(psi.norm))
print()

Hamiltonian = ising_impo(J, g)
t1 = time.time()
result = minimize(energy, x0=params, args=(c, Hamiltonian), method='nelder-mead')
sweet_spot = result.x
t2 = time.time()

print("goddamn, that took {}s to optimize".format(t2-t1))
print("sweet spot = {}".format(sweet_spot))
print("num function evaluations: {}".format(result['nfev']))
print("num iterations: {}".format(result['nit']))
print("termination msg: {}".format(result['message']))

from scipy.integrate import quad

def infinite_gs_energy(J, g):
    """
    Straight from tenpy docs: https://tenpy.readthedocs.io/en/latest/intro/examples/tfi_exact.html
    """
    def f(k, lambda_):
        return np.sqrt(1 + lambda_**2 + 2 * lambda_ * np.cos(k))

    E0_exact = -g / (J * 2. * np.pi) * quad(f, -np.pi, np.pi, args=(J / g, ))[0]
    return E0_exact

holo_gs = circuit_imps(sweet_spot, c)
holo_E = energy(sweet_spot, c, Hamiltonian)
E0 = infinite_gs_energy(J, g)
print("holoMPS: E = {:.8}".format(holo_E))
print(" theory: E = {:.8}".format(E0))
print("  error: {:.7%}".format(np.abs(1 - holo_E/E0)))