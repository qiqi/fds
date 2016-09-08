import copy

import numpy as np

from .op_base import OpBase

__all__ = ['transpose', 'reshape']

class transpose(OpBase):
    def __init__(self, a, axes=None):
        if axes is None:
            axes = tuple(reversed(range(a.ndim)))
        self.axes = tuple(axes)
        def func(x):
            if x.ndim == len(self.axes):
                return x.transpose(self.axes)
            else:
                return x.transpose(self.axes + (-1,))
        OpBase.__init__(self, func, (a,), name='transpose')

class reshape(OpBase):
    def __init__(self, a, shape):
        self.shape = np.empty(a.shape).reshape(shape).shape
        def func(x):
            if x.size == np.prod(self.shape):
                return x.reshape(self.shape)
            else:
                return x.reshape(self.shape + (-1,))
        OpBase.__init__(self, func, (a,), name='reshape')
