import copy

import numpy as np

from .op_base import OpBase

__all__ = ['getitem', 'setitem']

class getitem(OpBase):
    def __init__(self, a, ind):
        self.ind = copy.copy(ind)
        OpBase.__init__(self, lambda x: x[ind], (a,),
                        name='getitem[{0}]'.format(ind))

class setitem(OpBase):
    def __init__(self, a, ind, b):
        self.ind = copy.copy(ind)
        def op(x, a):
            x = copy.copy(x)
            x[self.ind] = a
            return x
        OpBase.__init__(self, op, (a, b), output_shapes=(a.shape,),
                        name='setitem[{0}]'.format(ind))
