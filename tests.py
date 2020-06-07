import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from circuit import Circuit
from arbitrary_SU4 import *

def check_unitary(u):
    udagu = np.matmul(jnp.conj(u).T, u)
    eye = np.eye(u.shape[0])
    assert np.allclose(udagu, eye, atol=1e-5)

def circuit_tests():
    print("TEST: Unitarity")
    c = Circuit([("qubit", "p", 2),
                ("cavity", "b", 3),
                ("cavity", "p", 4),
                ("qubit", "b", 2)])
    for i in range(30):
        rnd = np.random.randint(1, 9)
        if rnd == 1:
            c.add_gate("rotation", qids=[0])
        if rnd == 2:
            c.add_gate("displacement", qids=[1])
        if rnd == 3:
            c.add_gate("snap", qids=[0, 1])
        if rnd == 4:
            c.add_gate("snap", qids=[0, 2])
        if rnd == 5:
            c.add_gate("rotation", qids=[3])
        if rnd == 6:
            c.add_gate("displacement", qids=[2])
        if rnd == 7:
            c.add_gate("snap", qids=[3, 1])
        if rnd == 8:
            c.add_gate("snap", qids=[3, 2])
    c.assemble()
    params = np.random.rand(c.n_params)*10-5
    check_unitary(c.evaluate(params))
    print("PASS\n")

    print("TEST: Identity circuit")
    params = np.zeros(c.n_params)
    eye = np.eye(c.regInfo.dim)
    assert np.allclose(c.evaluate(params), eye, atol=1e-5)
    print("PASS\n")

    print("TEST: Tensor indexing 1")
    tensor = c.get_tensor(params)
    assert tensor.shape == (8, 6, 8, 6)
    print("PASS\n")

    print("TEST: Tensor indexing 2")
    c = Circuit([("qubit", "p", 2),
                ("cavity", "b", 10)])
    c.add_gate("displacement")
    c.add_gate("snap")
    c.add_gate("displacement")
    c.add_gate("snap")
    c.assemble()
    params = np.random.rand(c.n_params)*10-5
    tensor = c.get_tensor(params)
    assert (tensor[0, :, 1, :] == np.zeros((10, 10))).all()
    assert (tensor[1, :, 0, :] == np.zeros((10, 10))).all()
    print("PASS\n")

    print("TEST: Fock state creation")
    c = Circuit([("qubit", "p", 2),
                ("cavity", "b", 10)])
    c.add_gate("displacement")
    p1 = np.array([1.14, 0])
    c.add_gate("snap")
    p2 = np.zeros(10)
    p2[0] = np.pi
    c.add_gate("displacement")
    p3 = np.array([-0.58, 0])
    c.assemble()
    params = np.concatenate((p1, p2, p3))
    state = np.zeros(20)
    state[0] = 1
    final = c.evaluate(params) @ state
    assert np.linalg.norm(final) - 1 < 1e-5
    assert np.abs(final[1]) > 0.99
    print("PASS\n")

def arbitrary_su4_tests():
    rng = np.random.default_rng() 

    paulix = np.array([[0, 1], [1, 0]])
    pauliy = np.array([[0, -1j], [1j, 0]])
    pauliz = np.array([[1, 0], [0, -1]])

    print("TEST: cav_xrot")
    c = Circuit([("cavity", 'p', 2)])
    c.add_gate("displacement", n_params=1, fn=cav_xrot)
    c.assemble()
    for i in range(10):
        theta = rng.uniform(high=2*np.pi)
        u = c.evaluate(np.array([theta]))
        assert np.allclose(u, expm(-1j * theta/2 * paulix), atol=1e-5)
    print("PASS\n")
    
    print("TEST: cav_yrot")
    c = Circuit([("cavity", 'p', 2)])
    c.add_gate("displacement", n_params=1, fn=cav_yrot)
    c.assemble()
    for i in range(10):
        theta = rng.uniform(high=2*np.pi)
        u = c.evaluate(np.array([theta]))
        assert np.allclose(u, expm(-1j * theta/2 * pauliy), atol=1e-5)
    
    zx = expm(-1j * np.pi/4 * pauliy)
    zy = expm(-1j * np.pi/4 * paulix)
    eye = np.eye(2)
    print("PASS\n")

    print("TEST: cav_zx")
    c = Circuit([("cavity", 'p', 2)])
    c.add_gate("displacement", n_params=0, fn=cav_zx)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, zx, atol=1e-5)
    print("PASS\n")

    print("TEST: cav_xz")
    c = Circuit([("cavity", 'p', 2)])
    c.add_gate("displacement", n_params=0, fn=cav_zx)
    c.add_gate("displacement", n_params=0, fn=cav_xz)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, eye, atol=1e-5)
    print("PASS\n")

    print("TEST: cav_zy")
    c = Circuit([("cavity", 'p', 2)])
    c.add_gate("displacement", n_params=0, fn=cav_zy)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, zy, atol=1e-5)
    print("PASS\n")

    print("TEST: cav_yz")
    c = Circuit([("cavity", 'p', 2)])
    c.add_gate("displacement", n_params=0, fn=cav_zy)
    c.add_gate("displacement", n_params=0, fn=cav_yz)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, eye, atol=1e-5)
    print("PASS\n")

    print("TEST: qub_zx")
    c = Circuit([("qubit", 'p', 2)])
    c.add_gate("rotation", n_params=0, fn=qub_zx)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, zx, atol=1e-5)
    print("PASS\n")

    print("TEST: qub_xz")
    c = Circuit([("qubit", 'p', 2)])
    c.add_gate("rotation", n_params=0, fn=qub_zx)
    c.add_gate("rotation", n_params=0, fn=qub_xz)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, eye, atol=1e-5)
    print("PASS\n")

    print("TEST: qub_zy")
    c = Circuit([("qubit", 'p', 2)])
    c.add_gate("rotation", n_params=0, fn=qub_zy)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, zy, atol=1e-5)
    print("PASS\n")

    print("TEST: qub_yz")
    c = Circuit([("qubit", 'p', 2)])
    c.add_gate("rotation", n_params=0, fn=qub_zy)
    c.add_gate("rotation", n_params=0, fn=qub_yz)
    c.assemble()
    u = c.evaluate(np.array([]))
    assert np.allclose(u, eye, atol=1e-5)
    print("PASS\n")

    print("TEST: snap_zz")
    c = Circuit([("qubit", 'p', 2),
                 ("cavity", 'b', 2)])
    c.add_gate("snap", n_params=1, fn=snap_zz)
    c.assemble()
    zz = np.kron(pauliz, pauliz)
    for i in range(10):
        theta = rng.uniform(-10, 10)
        u = c.evaluate(np.array([theta]))
        ans = expm(1j * theta * zz)
        assert np.allclose(u, ans, atol=1e-5)
    print("PASS\n")


if __name__ == "__main__":
    print("=============")
    print("CIRCUIT TESTS")
    print("=============")
    circuit_tests()
    print("=====================")
    print("ARBITRARY SU(4) TESTS")
    print("=====================")
    arbitrary_su4_tests()
