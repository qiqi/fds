import numpy as np

from .op_base import OpBase

class QRT(OpBase):
    def __init__(self, A):
        def qr(A):
            Q, R = np.linalg.qr(A.T)
            return Q.T, R
        output_shapes=(A.shape, A.shape + A.shape)
        OpBase.__init__(self, qr, (A,),
                        are_outputs_distributed=(True, False),
                        name='qr', output_shapes=output_shapes)

class ReduceSum(OpBase):
    def __init__(self, x):
        assert x.is_distributed
        OpBase.__init__(self, lambda x : x.sum(-1), (x,),
                        output_shapes=(x.shape,),
                        are_outputs_distributed=(False,), name='ReduceSum')

class broadcast(OpBase):
    def __init__(self, x):
        assert not x.is_distributed
        OpBase.__init__(self, lambda x : x.reshape(x.shape + (1,)), (x,),
                        output_shapes=(x.shape,),
                        are_outputs_distributed=(True,), name='broadcast')

