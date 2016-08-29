from __future__ import division
import os
import sys
import shutil

import numpy as np

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..', '..'))

import pascal_lite as pascal

def test_add_mul():
# if __name__ == '__main__':
    subspace_dimension = 16

    V = pascal.random(subspace_dimension)
    V = V.reshape([1, -1]).T.transpose().ravel()
    v0 = pascal.ones().copy() + pascal.zeros()
    v1 = pascal.symbolic_array()
    v2 = pascal.symbolic_array(1)
    a = np.ones(subspace_dimension)

    V[1:2] = 1

    v3 = -v0 / 2 + (V * a).sum() - 2 * v1 + v2[0]
    v3 = 0.5 - (-v3) / 2
    v3 = -0.5 + v3
    v3 *= 2
    print(v0, v1, v2, len(v0))
    g = pascal.ComputationalGraph([v3.value])

    n_for_this_mpi_rank = 100000
    actual_inputs = {
            pascal.builtin.ZERO: np.zeros(n_for_this_mpi_rank),
            pascal.builtin.RANDOM[0]:
                np.ones(pascal.builtin.RANDOM[0].shape +
                        (n_for_this_mpi_rank,)),
            v1.value: np.ones(n_for_this_mpi_rank),
            v2.value: 2.5 * np.ones(n_for_this_mpi_rank)
            }

    actual_output, = g(actual_inputs)
    assert actual_output.shape == (n_for_this_mpi_rank,)
    assert abs(actual_output - subspace_dimension).max() < 1E-12

    # a different interface
    def actual_inputs(x):
        if x is pascal.builtin.ZERO:
            return np.zeros(n_for_this_mpi_rank)
        elif x is pascal.builtin.RANDOM[0]:
            return np.ones(pascal.builtin.RANDOM[0].shape +
                           (n_for_this_mpi_rank,))
        elif x is v1.value:
            return np.ones(n_for_this_mpi_rank)
        elif x is v2.value:
            return 2.5 * np.ones(n_for_this_mpi_rank)

    actual_output, = g(actual_inputs)
    assert actual_output.shape == (n_for_this_mpi_rank,)
    assert abs(actual_output - subspace_dimension).max() < 1E-12

def test_linalg():
#if __name__ == '__main__':
    subspace_dimension = 4
    V = pascal.symbolic_array(subspace_dimension)
    v = pascal.symbolic_array()

    v1 = pascal.dot(V, v)
    v2 = pascal.outer(v1, v)
    v3 = pascal.qr_transpose(V)[0]

    g = pascal.ComputationalGraph([v1.value, v2.value, v3.value])
    n = 16

    A = np.random.rand(n, subspace_dimension)
    b = np.random.rand(subspace_dimension)

    def actual_inputs(x):
        if x is V.value:
            return A
        elif x is v.value:
            return b

    o1, o2, o3 = g(actual_inputs)
    assert np.allclose(o1, np.dot(A, b))
    assert np.allclose(o2, np.outer(np.dot(A, b), b))
    assert np.allclose(o3, np.linalg.qr(A.T)[0].T)

