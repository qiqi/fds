import copy
import operator
import numbers

import numpy as np

from .op_base import OpBase, OpBase, OpBase, OpBase

__all__ = ['add', 'sub', 'mul', 'truediv', 'neg', 'pow', 'sum']

class add(OpBase):
    def __init__(self, a, b):
        OpBase.__init__(self, operator.add, (a, b), name="add")

class sub(OpBase):
    def __init__(self, a, b):
        OpBase.__init__(self, operator.sub, (a, b), name="sub")

class mul(OpBase):
    def __init__(self, a, b):
        OpBase.__init__(self, operator.mul, (a, b), name="mul")

class truediv(OpBase):
    def __init__(self, a, b):
        OpBase.__init__(self, operator.truediv, (a, b), name="div")

class neg(OpBase):
    def __init__(self, a):
        OpBase.__init__(self, operator.neg, (a,), name="neg")

class pow(OpBase):
    def __init__(self, a, b):
        OpBase.__init__(self, operator.pow, (a, b), name="pow")

class sum(OpBase):
    def __init__(self, a, axis=None):
        self.axis = copy.copy(axis)
        OpBase.__init__(self, lambda x: x.sum(self.axis),
                        (a,), name='sum')
