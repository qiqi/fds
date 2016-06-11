from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

class LssTangent:
    def __init__(self):
        self.Rs = []
        self.bs = []

    def m_segments(self):
        assert len(self.Rs) == len(self.bs)
        return len(self.Rs)

    def K_modes(self):
        return self.Rs[0].shape[0]

    def checkpoint(self, V, v):
        Q, R = linalg.qr(V)
        b = dot(Q.T, v)
        self.Rs.append(R)
        self.bs.append(b)
        V[:] = Q
        v -= dot(Q, b)

    def solve(self):
        Rs, bs = array(self.Rs), array(self.bs)
        assert Rs.ndim == 3 and bs.ndim == 2
        assert Rs.shape[0] == bs.shape[0]
        assert Rs.shape[1] == Rs.shape[2] == bs.shape[1]
        nseg, subdim = bs.shape
        eyes = eye(subdim, subdim) * ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, r_[1:nseg+1], r_[:nseg+1]))
        D = sparse.bsr_matrix((Rs, r_[:nseg], r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        alpha = -(B.T * splinalg.spsolve(B * B.T, ravel(bs)))
        return alpha.reshape([nseg+1,-1])[:-1]

