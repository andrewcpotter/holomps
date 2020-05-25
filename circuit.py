import jax.numpy as np
from jax.scipy.linalg import expm
from jax import jit
import numpy as onp
from functools import partial

class Circuit:
    """
    Represents a quantum circuit.
    
    Attributes
    ----------
    regInfo: RegisterInfo
        Contains information about the dimension, shape, etc of the qudit register.
    gates: list
        The quantum gates comprising the circuit. After assembly, this consists of GateLayers
        which themselves contain the actual gates.
    n_params: int
        The number of gate parameters in the circuit. Only set after the circuit is assembled.
    """
    
    def __init__(self, register):
        """
        Parameters
        ----------
        register: list
            A list of ints describing the number of levels in each qudit. len(register) should
            equal the number of qudits, with register[i] giving the number of levels in the
            i^th qudit (e.g. 2 for a qubit).            
        """
        self.regInfo = RegisterInfo(register)
        self.gates = []
        self.n_params = None
        
    def add_gate(self, gate_type, qids):
        """
        Add a gate to the circuit. Note that the ordering of qids is important for
        multi-qudit gates. For a SNAP gate, the first qudit id refers to the qubit,
        and the second refers to the cavity.
        
        Parameters
        ----------
        gate_type: str
            One of "rotation", "displacement", or "snap".
        qids: tuple or int
            The qudit labels that this gate acts on. Qudits are labelled by their index in
            the register (starting at 0). If qids=(1,3,4), then this gate acts on qudits
            1, 3, and 4.
        """
        if type(qids) is int:
            qids = (qids,)
        for qid in qids:
            assert qid < len(self.regInfo.register), "Invalid qudit_id"
        gate_type = gate_type.lower()
        gate = None
        if gate_type == "rotation":
            assert len(qids) == 1, "Rotation gate must act on exactly 1 qubit"
            qid = qids[0]
            assert self.regInfo.register[qid] == 2, "Rotation qubit should have 2 levels"
            gate = RotGate(qid)
        elif gate_type == "displacement":
            assert len(qids) == 1, "Displacement gate must act on exactly 1 qudit"
            qid = qids[0]
            dim = self.regInfo.register[qid]
            assert dim >= 3, "Cavity should have at least 3 levels"
            gate = DispGate(qid, dim)
        elif gate_type == "snap":
            assert len(qids) == 2, "SNAP gate must act on exactly 2 qudits"
            qubit_id, cavity_id = qids
            assert self.regInfo.register[qubit_id] == 2, "SNAP qubit should have 2 levels"
            assert self.regInfo.register[cavity_id] >= 3, "SNAP cavity should have at least 3 levels"
            dim = self.regInfo.register[qubit_id] * self.regInfo.register[cavity_id]
            gate = SNAPGate(qubit_id, cavity_id, dim)
        else:
            assert False, "Invalid gate type."
        if gate is not None:
            self.gates.append(gate)

    def assemble(self):
        """
        Assemble the circuit to make it ready for evaluation. Organizes the gates into
        layers using a simple non-overlap scheme, assigns each gate a region of the
        parameter vector, and fills "incomplete" layers with identity gates. Note that
        after assembly, the 
        """
        # fill out the current layer and append it to the list of layers.
        def complete(layer):
            # use symmetric difference of sets to find unused qudits in this layer
            unused = used_ids ^ set(self.regInfo.ids)
            # fill the layer with identity gates for each unused qudit
            for qid in unused:
                dim = self.regInfo.register[qid]
                layer.append(Gate(dim, [qid], 0))
            layers.append(GateLayer(gates=layer, regInfo=self.regInfo))
        
        layers = []
        # a counter that represents the next gate's index in the parameter vector
        param_ind = 0
        # all the gates that are going into the current layer
        layer = []
        # all the qudit ids that are being used in the current layer
        used_ids = set()
        for g in self.gates:
            g.param_ind = param_ind
            param_ind += g.n_params
            g_ids = set(g.qudit_ids)
            # check if the gate overlaps with any of the gates in the current layer
            # if so, fill out this layer, append it, and start a new layer
            if len(g_ids.intersection(used_ids)) != 0:
                complete(layer)
                used_ids = set()
                layer = []
            layer.append(g)
            used_ids = used_ids.union(g_ids)
        # take care of the last layer
        complete(layer)
        
        # gates are now the series of GateLayers
        self.gates = layers
        self.n_params = param_ind

    @partial(jit, static_argnums=(0,))
    def evaluate(self, params):
        """
        Evaluates the circuit and returns its unitary operator. You must assemble the
        circuit before evaluating it. Uses a JIT compiler; for optimal performance, do not
        modify the circuit nor its constituent gates between calls to this method.
        
        Parameters
        ----------
        params: 1D ndarray
            A vector of parameters for the gates.
        
        Returns
        -------
        2D ndarray. A unitary matrix representing the circuit.
        """
        assert len(params) == self.n_params
        mat = np.eye(self.regInfo.dim)
        for layer in self.gates:
            g = layer.gate(params)
            # reverse since operators act right-to-left
            mat = np.matmul(g, mat)
        return mat

class RegisterInfo:
    """
    Information about a qudit register.
    
    Attributes
    ----------
    register: list
        A list of ints describing the number of levels in each qudit. len(register) should
        equal the number of qudits, with register[i] giving the number of levels in the
        i^th qudit (e.g. 2 for a qubit).
    dim: int
        The total Hilbert dimension of the register
    shape: list
        The qudit-indexed shape of a unitary tensor acting on the register
    ids: iterator
        The labels for each of the qudits. Given by range(len(register))
    """
    
    def __init__(self, register):
        self.register = list(register)
        self.dim = np.prod(register)
        self.shape = self.register + self.register
        self.ids = range(len(self.register))

class Gate:
    """
    Base class for all gates. By default, implements the identity matrix.
    The gate method returns the unitary matrix corresponding to this gate. All
    subclasses should override this method, and implementations should be as efficient
    as possible.
    """
    
    def __init__(self, dim, qids, n_params=0):
        """
        Parameters
        ----------
        dim: int
            The dimension of this gate. Should be the product of local dimensions for all
            qudits that this gate acts on.
        qids: list
            A list of the qudit labels (in the parent circuit) that this gate acts on
        n_params: int
            The number of parameters this gate requires
        """
        self.dim = dim
        self.qudit_ids = qids
        self.n_params = n_params
        # param_ind: where in the parameter vector to look for relevant params
        # this is currently set during circuit assembly
        self.param_ind = None
        
    def gate(self, params):
        """
        Parameters
        ----------
        params: 1D ndarray
            The full parameter vector. Each gate should use its param_ind to locate the
            relevant parameters.
        """
        return np.eye(self.dim)

class RotGate(Gate):
    
    def __init__(self, qubit_id):
        super().__init__(dim=2, qids=[qubit_id], n_params=3)
        self.paulix = np.array([[0, 1],[1, 0]])
        self.pauliy = np.array([[0,-1j],[1j, 0]])
        self.pauliz = np.array([[1, 0],[0,-1]])

    def gate(self, params):
        # get relevant params from param vector
        ind = self.param_ind
        theta, phi, rotangle = params[ind:ind+self.n_params]
        # use cayley-hamilton theorem for variant form of rotation operator
        # https://arxiv.org/pdf/1402.3541.pdf
        inS = 1j * (np.sin(theta)*np.cos(phi)*self.paulix + \
                    np.sin(theta)*np.sin(phi)*self.pauliy + \
                    np.cos(theta)*self.pauliz)
        return np.cos(rotangle/2)*np.eye(2) + np.sin(rotangle/2)*inS
    
class DispGate(Gate):
    
    def __init__(self, cavity_id, dim):
        super().__init__(dim, qids=[cavity_id], n_params=2)
        # hard code the creation and annihilation operators
        self.cr = onp.zeros((self.dim, self.dim))
        self.an = onp.zeros((self.dim, self.dim))
        for i in range(self.dim-1):
            self.cr[i+1, i] = onp.sqrt(i+1)
            self.an[i, i+1] = onp.sqrt(i+1)
    
    def gate(self, params):
        ind = self.param_ind
        a, b = params[ind:ind+self.n_params]
        alpha = a + b*1j
        astar = alpha.conjugate()
        return expm(alpha*self.cr - astar*self.an)

class SNAPGate(Gate):
    
    def __init__(self, qubit_id, cavity_id, dim):
        qids = [qubit_id, cavity_id]
        super().__init__(dim, qids, n_params=dim//2)
    
    def gate(self, params):
        ind = self.param_ind
        theta = params[ind:ind+self.n_params]
        diag = np.exp(np.concatenate((1j*theta, -1j*theta)))
        return np.diag(diag)

class GateLayer(Gate):
    
    def __init__(self, gates, regInfo):
        # try to order gates by qudit to minimize permutation during gate calculation
        self.gates = sorted(gates, key=lambda x: x.qudit_ids[0])
        # get the order of qids so that we know how to permute if necessary
        qids = []
        for g in self.gates:
            qids += g.qudit_ids
        # make sure this layer doesn't have any overlapping gates
        assert len(set(qids)) == len(qids)
        
        self.regInfo = regInfo
        self.permuted = qids
        super().__init__(self.regInfo.dim, qids)
    
    def gate(self, params):
        # start with identity scalar and tensor-product all the gates
        mat = 1
        for g in self.gates:
            g = g.gate(params)
            mat = np.kron(mat, g)
        # if we have interleaved gates (i.e. a gate acting on qudit #1 and #3 but not #2)
        # then we need to un-permute the indices
        # NOTE: this hasn't been tested so idk if this is right
        if self.permuted != self.regInfo.ids:
            # reshape 2D unitary into qudit indices
            mat = mat.reshape(self.regInfo.shape)
            mat = np.moveaxis(mat, self.permuted, self.regInfo.ids)
            mat = mat.reshape((self.regInfo.dim, self.regInfo.dim))
        return mat