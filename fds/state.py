import numpy as np

class FromFile:
    def __init__(self, fname):
        self.fname = fname

class Random:
    generator = np.random.random

class Zeros:
    generator = np.zeros

class SymbolicState:
    def __init__(self, parent):
        self.parent = parent

    def __array__(self, dtype=float):
        return self.ndarray

class SymbolicStateArray:
    def __init__(self, states):
        self.states = states

    def __array__(self, dtype=float):
        return np.array([np.array(s) for s in self.states])

def encode_state(u):
    if isinstance(u, SymbolicState):
        return u
    elif isinstance(u, str):
        return SymbolicState(FromFile(u))
    else:
        return np.array(u, dtype=float)

def decode_state(u):
    if isinstance(u, np.ndarray):
        return u
    elif isinstance(u, str):
        return u
    else:
        raise NotImplementedError

def random_state():
    return SymbolicState(Random())

def random_states(n):
    return SymbolicStateArray([random_state() for i in range(n)])

def zero_state():
    return SymbolicState(Zeros())

def try_convert_to_ndarray(u, size):
    u.ndarray = u.parent.generator(size)
    u.parent = None

def state_dot(u, v):
    if isinstance(u, SymbolicStateArray):
        return [state_dot(ui, v) for ui in u.states]
    elif isinstance(u, SymbolicState) and hasattr(u.parent, 'generator'):
        if isinstance(v, np.ndarray):
            try_convert_to_ndarray(u, v.shape[0])
    return np.dot(u, v)

def state_outer(u, v):
    return np.outer(u, v)

def state_norm(u):
    return np.linalg.norm(u)

def qr_transpose_states(V):
    Q, R = np.linalg.qr(np.transpose(V))
    return Q.T, R
