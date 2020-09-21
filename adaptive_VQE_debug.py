from circuit_qubit import *

def qub_z(params): return (0,0,params[0])
def qub_two(params): return (params[0])

d = 2
chimax = 2

Entangler_pool = [XX_YYGate(qids = [0,1], fn = qub_two),XX_YYGate(qids = [0,2], fn=qub_two),XX_YYGate(qids = [0,3], fn=qub_two),\
XX_YYGate(qids = [1,2], fn = qub_two),XX_YYGate(qids = [1,3], fn = qub_two),XX_YYGate(qids = [2,3], fn = qub_two),\
ZZGate(qids = [0,1], fn = qub_two), ZZGate(qids = [0,2], fn = qub_two), ZZGate(qids = [0,3], fn = qub_two),\
ZZGate(qids = [1,2], fn = qub_two), ZZGate(qids = [1,3], fn = qub_two), ZZGate(qids = [2,3], fn = qub_two),\
RotGate(qids=[0], n_params=1, fn=qub_z), RotGate(qids=[1], n_params=1, fn=qub_z),\
RotGate(qids=[2], n_params=1, fn=qub_z), RotGate(qids=[3], n_params=1, fn=qub_z)]

gate_choice = RotGate(qids=[2], n_params=1, fn=qub_z)

c0 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
c1 = Circuit([("qubit", "p", d), ("qubit", "p", d), ("qubit", "b", chimax), ("qubit", "b", chimax)])
c0.gates.append(gate_choice)
c0.gates.append(gate_choice)
c00 = c0
c0.assemble()
c00.gates.append(gate_choice)
c00.assemble()
print(c00.n_params)

c0.gates.append(gate_choice)
c0.assemble()
print(c0.n_params)

def energy_part_vary(params, known_params, c, c1, gate, gate1, Hamiltonian):
    params_total = known_params[0, c.n_params] + params[0,gate.n_params] + known_params[c.n_params, c.n_params+c1.n_params] +\
    params[gate.n_params, gate.n_params  + gate1.n_params]
    c.gates.append(gate)
    c1.gates.append(gate1)
    psi = circuit_imps(params_total, c, c1)
    E = (Hamiltonian.expectation_value(psi)).real
    filling_now = psi.expectation_value("Ntot")
    filling_diff = abs(filling_now[0] + filling_now[1] - 2)
    E_fix_filling = E + filling_diff * 10
    return E_fix_filling

