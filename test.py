import jax.numpy as jnp
import numpy as np
from circuit import Circuit

def check_unitary(u):
    udagu = np.matmul(jnp.conj(u).T, u)
    imat = np.eye(u.shape[0])
    diff = udagu - imat
    assert (diff.conj() * diff).sum() < 1e-10*u.shape[0]

if __name__ == "__main__":
    print("TESTS\n----------\n")
    print("Test: Unitarity")
    for trial in range(10):
        c = Circuit([2, 10])
        for i in range(10):
            rnd = np.random.randint(1, 4)
            if rnd == 1:
                c.add_gate("rotation", 0)
            if rnd == 2:
                c.add_gate("displacement", 1)
            if rnd == 3:
                c.add_gate("snap", (0, 1))
        c.assemble()
        params = np.random.rand(c.n_params)*10-5
        check_unitary(c.evaluate(params))
    print("Pass\n")


    print("Test: Identity circuit")
    params = np.zeros(c.n_params)
    assert (np.eye(c.regInfo.dim) == c.evaluate(params)).all()
    print("Pass\n")

    print("Test: Fock state creation")
    c = Circuit([2, 10])
    c.add_gate("displacement", 1)
    p1 = np.array([1.14, 0])
    c.add_gate("snap", [0, 1])
    p2 = np.zeros(10)
    p2[0] = np.pi
    c.add_gate("displacement", 1)
    p3 = np.array([-0.58, 0])
    c.assemble()
    params = np.concatenate((p1, p2, p3))
    state = np.zeros(20)
    state[0] = 1
    final = c.evaluate(params) @ state
    assert np.linalg.norm(final) - 1 < 1e-5
    assert np.abs(final[1]) > 0.99
    print("Pass\n")
