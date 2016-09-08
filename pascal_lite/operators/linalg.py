import numpy as np

from .op_base import OpBase

try:    
    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    if mpi.Get_size() == 1:
        mpi = None
except ImportError:
    mpi = None

from .plinalg import pQR, pdot

class QRT(OpBase):
    def __init__(self, A):
        def qr(A):
            if mpi:
                Q, R = pQR(mpi, A.T)
            else:
                Q, R = np.linalg.qr(A.T)
            return Q.T, R
        output_shapes=(A.shape, A.shape + A.shape)
        OpBase.__init__(self, qr, (A,),
                        are_outputs_distributed=(True, False),
                        name='qr', output_shapes=output_shapes)

class Dot(OpBase):
    def __init__(self, x, y):
        assert x.is_distributed
        assert y.is_distributed
        def dot(x, y):
            if mpi:
                return pdot(mpi, x, y)
            else:
                return (x * y).sum(-1)
        OpBase.__init__(self, dot, (x,y),
                        output_shapes=(x.shape,),
                        are_outputs_distributed=(False,), name='ReduceSum')

class broadcast(OpBase):
    def __init__(self, x):
        assert not x.is_distributed
        OpBase.__init__(self, lambda x : x.reshape(x.shape + (1,)), (x,),
                        output_shapes=(x.shape,),
                        are_outputs_distributed=(True,), name='broadcast')

