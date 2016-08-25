import numpy as np

from .op_base import OpBase

class QRT(OpBase):
    def __init__(self, A, **kwargs):
        def qr(A):
            Q, R = np.linalg.qr(A.T)
            return Q.T, R.T
        OpBase.__init__(self, qr, (A,), name='qr', **kwargs)

class Dot(OpBase):
    pass
