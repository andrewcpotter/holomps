import numpy as np
from circuit import Circuit

"""
These function definitions are used to restrict the behavior of gates we use.
For example, the rotation gate is natively an arbitrary rotation around an
arbitrary axis, but if we supply the gate with the qub_zx function, then we
restrict the gate to a pi/2 rotation around the Y axis (no free parameters).
See documentation for Circuit.add_gate, RotGate, DispGate, and SNAPGate
for details
"""
# treat the cavity as a qubit and do an arbitrary X rotation
def cav_xrot(params):
    theta = params[0]
    return (0, -theta/2)
# treat the cavity as a qubit and do an arbitrary Y rotation
def cav_yrot(params):
    theta = params[0]
    return (theta/2, 0)
# treat the cavity as a qubit and rotate from Z to X
def cav_zx(params): return (np.pi/4, 0)
# treat the cavity as a qubit and rotate from X to Z
def cav_xz(params): return (-np.pi/4, 0)
# treat the cavity as a qubit and rotate from Z to Y
def cav_zy(params): return (0, -np.pi/4)
# treat the cavity as a qubit and rotate from Y to Z
def cav_yz(params): return (0, np.pi/4)
# rotate qubit from Z to X
def qub_zx(params): return (np.pi/2, np.pi/2, np.pi/2)
# rotate qubit from X to Z
def qub_xz(params): return (np.pi/2, np.pi/2, -np.pi/2)
# rotate qubit from Z to Y
def qub_zy(params): return (np.pi/2, 0, np.pi/2)
# rotate qubit from Y to Z
def qub_yz(params): return (np.pi/2, 0, -np.pi/2)
# SNAP equivalent of exp(i theta ZZ)
def snap_zz(params):
    theta = params[0]
    return [theta, -theta]

if __name__ == "__main__":
    c = Circuit([("qubit", "p", 2),
                 ("cavity", "b", 2)])

    # arbitrary one-qubit rotations
    c.add_gate("rotation")
    c.add_gate("displacement", n_params=1, fn=cav_xrot)
    c.add_gate("displacement", n_params=1, fn=cav_yrot)
    c.add_gate("displacement", n_params=1, fn=cav_xrot)

    # XX rotation
    c.add_gate("rotation", n_params=0, fn=qub_xz)
    c.add_gate("displacement", n_params=0, fn=cav_xz)
    c.add_gate("snap", n_params=1, fn=snap_zz)
    c.add_gate("rotation", n_params=0, fn=qub_zx)
    c.add_gate("displacement", n_params=0, fn=cav_zx)

    # YY rotation
    c.add_gate("rotation", n_params=0, fn=qub_yz)
    c.add_gate("displacement", n_params=0, fn=cav_yz)
    c.add_gate("snap", n_params=1, fn=snap_zz)
    c.add_gate("rotation", n_params=0, fn=qub_zy)
    c.add_gate("displacement", n_params=0, fn=cav_zy)

    # ZZ rotation
    c.add_gate("snap", n_params=1, fn=snap_zz)

    # arbitrary one-qubit rotations
    c.add_gate("rotation")
    c.add_gate("displacement", n_params=1, fn=cav_xrot)
    c.add_gate("displacement", n_params=1, fn=cav_yrot)
    c.add_gate("displacement", n_params=1, fn=cav_xrot)


    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    c.assemble()
    print()
    print("num free params: {}".format(c.n_params))
    print()

    params = np.random.rand(c.n_params)*10-5
    print("param vector:\n{}".format(params))
    print()

    u = c.evaluate(params)
    print("resulting unitary:\n{}".format(u))
    print()

    tensor = c.get_tensor(params)
    print("resulting tensor:\n{}".format(tensor))
    print()
    print("tensor shape:\n{}".format(tensor.shape))
