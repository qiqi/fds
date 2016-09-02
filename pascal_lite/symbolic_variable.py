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

def _binary_op(a, b, op):
    is_a_like_sa = _is_like_sa(a)
    is_b_like_sa = _is_like_sa(b)
    is_result_distributed = (is_a_like_sa and a.is_distributed or
                             is_b_like_sa and b.is_distributed)
    if is_result_distributed:
        if is_a_like_sa and not a.is_distributed:
            a = broadcast(a)
        elif is_b_like_sa and not b.is_distributed:
            b = broadcast(b)
    a = a.value if is_a_like_sa else a
    b = b.value if is_b_like_sa else b
    return symbolic_array(op(a, b).output)


# ============================================================================ #
#                          symbolic array variable                             #
# ============================================================================ #

class symbolic_array(object):

    __context__ = sys.modules[__name__]

    def __init__(self, init=(), field=None, is_distributed=True):
        if _is_like_sa_value(init):
            self.value = init
        else:
            shape = init
            if isinstance(shape, int):
                shape = (shape,)
            self.value = symbolic_array_value(shape, field=field,
                                              is_distributed=is_distributed)

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

    @property
    def is_distributed(self):
        return self.value.is_distributed

    def __len__(self):
        return len(self.value)

    # --------------------------- operations ------------------------------ #

    # asks ndarray to use the __rops__ defined in this class
    __array_priority__ = 3000

    def __add__(self, a):
        return _binary_op(self, a, operators.add)

    def __radd__(self, a):
        return self.__add__(a)

    def __iadd__(self, a):
        self[:] = self + a
        return self

    def __sub__(self, a):
        return _binary_op(self, a, operators.sub)

    def __rsub__(self, a):
        return _binary_op(a, self, operators.sub)

    def __isub__(self, a):
        self[:] = self - a
        return self

    def __mul__(self, a):
        return _binary_op(self, a, operators.mul)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        self[:] = self * a
        return self

    def __truediv__(self, a):
        return _binary_op(self, a, operators.truediv)

    def __idiv__(self, a):
        self[:] = self / a

    def __neg__(self):
        return symbolic_array(operators.neg(self.value).output)

    def __pow__(self, power):
        return _binary_op(self, power, operators.pow)

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

def broadcast(x):
    return symbolic_array(operators.broadcast(x.value).output)

def qr_transpose(x):
    assert _is_like_sa(x)
    qr_op = operators.QRT(x.value)
    outputs = tuple([symbolic_array(y) for y in qr_op.outputs])
    return outputs

def dot(x, y):
    if (_is_like_sa(x) and _is_like_sa(y) and x.is_distributed
                                          and y.is_distributed):
        return symbolic_array(operators.Dot(x.value, y.value).output)
    else:
        return (x * y).sum()

def norm(x):
    return dot(x, x)**0.5

def outer(x, y):
    x = x.reshape(x.shape + (1,) * y.ndim)
    y = y.reshape((1,) * x.ndim + y.shape)
    return x * y

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
    array = symbolic_array(builtin.ZERO)
    array.value.field = 0
    return array + np.zeros(shape)

def random(shape=()):
    array = symbolic_array(random_value(shape))
    array.value.field = 1
    return array


################################################################################
################################################################################
################################################################################
