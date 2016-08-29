import numpy as np

from .op_base import OpBase

class QRT(OpBase):
    def __init__(self, A):
        def qr(A):
            Q, R = np.linalg.qr(A.T)
            return Q.T, R.T
        shapes=(A.shape, A.shape + A.shape)
        OpBase.__init__(self, qr, (A,), name='qr', shapes=shapes)

class Dot(OpBase):
    def __init__(self, x, y):
        shapes = (x.shape,)
        OpBase.__init__(self, np.dot, (x, y), name='dot')

class Outer(OpBase):
    def __init__(self, x, y):
        shapes = (x.shape,)
        OpBase.__init__(self, np.outer, (x, y), name='outer', shapes=shapes)
