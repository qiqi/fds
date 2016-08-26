################################################################################
#                                                                              #
#   symbolic_variable.py copyright(c) Qiqi Wang 2015 (qiqi.wang@gmail.com)     #
#                                                                              #
################################################################################

import sys

import numpy as np

from . import operators
from . import symbolic_value
from .symbolic_value import _is_like_sa_value, symbolic_array_value, \
                            random_value, builtin

__all__ = ['symbolic_array', 'transpose', 'reshape', 'copy', 'ravel',
           'sum', 'ones', 'zeros', 'random', 'qr_transpose', 'dot', 
           'norm', 'outer']

# ============================================================================ #

def _is_like_sa(a):
    '''
    Check if object has a symbolic array value
    '''
    return hasattr(a, 'value') and _is_like_sa_value(a.value)

# ============================================================================ #
#                          symbolic array variable                             #
# ============================================================================ #

class symbolic_array(object):

    __context__ = sys.modules[__name__]

    def __init__(self, init=(), field=None):
        if _is_like_sa_value(init):
            self.value = init
        else:
            shape = init
            if isinstance(shape, int):
                shape = (shape,)
            self.value = symbolic_array_value(shape, field=field)

    def __repr__(self):
        return 'Variable holding {0}'.format(self.value)

    # --------------------------- properties ------------------------------ #

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @property
    def field(self):
        return self.value.field

    def __len__(self):
        return len(self.value)

    # --------------------------- operations ------------------------------ #

    # asks ndarray to use the __rops__ defined in this class
    __array_priority__ = 3000

    def __add__(self, a):
        a = a.value if _is_like_sa(a) else a
        return symbolic_array(operators.add(self.value, a).output)

    def __radd__(self, a):
        return self.__add__(a)

    def __sub__(self, a):
        a = a.value if _is_like_sa(a) else a
        return symbolic_array(operators.sub(self.value, a).output)

    def __rsub__(self, a):
        a = a.value if _is_like_sa(a) else a
        return symbolic_array(operators.sub(a, self.value).output)

    def __mul__(self, a):
        a = a.value if _is_like_sa(a) else a
        return symbolic_array(operators.mul(self.value, a).output)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __truediv__(self, a):
        a = a.value if _is_like_sa(a) else a
        return symbolic_array(operators.truediv(self.value, a).output)

    def __neg__(self):
        return symbolic_array(operators.neg(self.value).output)

    def __pow__(self, power):
        return symbolic_array(operators.pow(self.value, power).output)

    # ------------------------- math functions ---------------------------- #

    def sum(self, axis=None):
        return sum(self, axis)

    # ------------------------- transformations --------------------------- #

    @property
    def T(self):
        return transpose(self)

    def transpose(self, axes=None):
        return transpose(self, axes)

    def reshape(self, shape):
        return reshape(self, shape)

    def copy(self):
        return copy(self)

    def ravel(self):
        return ravel(self)

    # ---------------------------- indexing ------------------------------- #

    def __getitem__(self, ind):
        return symbolic_array(operators.getitem(self.value, ind).output)

    def __setitem__(self, ind, a):
        a = a.value if _is_like_sa(a) else np.array(a, float)
        owner = operators.setitem(self.value, ind, a)
        assert self.shape == owner.output.shape
        self.value = owner.output


# ============================================================================ #
#                             data transformations                             #
# ============================================================================ #

def transpose(x, axes=None):
    assert _is_like_sa(x)
    return symbolic_array(operators.transpose(x.value, axes).output)

def reshape(x, shape):
    assert _is_like_sa(x)
    return symbolic_array(operators.reshape(x.value, shape).output)

def copy(x):
    assert _is_like_sa(x)
    return symbolic_array(x.value)

def ravel(x):
    return reshape(x, (x.size,))

def qr_transpose(x):
    assert _is_like_sa(x)
    outputs = operators.QRT(x.value)
    outputs = tuple([symbolic_array(y) for y in outputs.outputs])
    return outputs

def dot(x, y):
    assert _is_like_sa(x)
    assert _is_like_sa(y)
    return symbolic_array(operators.Dot(x.value, y.value).output)

def norm(x):
    return dot(x, x)**0.5

def outer(x, y):
    assert _is_like_sa(x)
    assert _is_like_sa(y)
    return symbolic_array(operators.Outer(x.value, y.value).output)

# ============================================================================ #
#                            mathematical functions                            #
# ============================================================================ #

def sum(a, axis=None):
    assert _is_like_sa(a)
    if axis is None:
        axis = tuple(range(a.ndim))
    return symbolic_array(operators.sum(a.value, axis).output)

# ============================================================================ #
#                             array generators                                 #
# ============================================================================ #

def ones(shape=()):
    return symbolic_array(builtin.ZERO) + np.ones(shape)

def zeros(shape=()):
    return symbolic_array(builtin.ZERO) + np.zeros(shape)

def random(shape=()):
    return symbolic_array(random_value(shape))


################################################################################
################################################################################
################################################################################
