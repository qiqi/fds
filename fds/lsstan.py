from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

class LssTangent:
    def __init__(self):
        self.Rs = []
        self.bs = []

    def K_segments(self):
        assert len(self.Rs) == len(self.bs)
        return len(self.Rs)

    def m_modes(self):
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
        # alpha1 = splinalg.lsqr(B, ravel(bs), iter_lim=10000)
        return alpha.reshape([nseg+1,-1])[:-1]

    def solve_dim(self, dim):
        Rs, bs = array(self.Rs), array(self.bs)
        Rs11, bs1 = Rs[:,dim:,dim:], bs[:,dim:]
        alpha = zeros_like(bs)
        for i in range(len(bs)-1):
            alpha[i+1,dim:] = dot(Rs11[i], alpha[i,dim:]) + bs1[i]
            bs[i] -= alpha[i+1] - dot(Rs[i], alpha[i])
        Rs00, bs0 = Rs[:,:dim,:dim], bs[:,:dim]
        nseg, subdim = bs0.shape
        eyes = eye(subdim, subdim) * ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, r_[1:nseg+1], r_[:nseg+1]))
        D = sparse.bsr_matrix((Rs00, r_[:nseg], r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        a = -(B.T * splinalg.spsolve(B * B.T, ravel(bs0)))
        alpha[:,:dim] = a.reshape([nseg+1,-1])[:-1]

    def lyapunov_exponents(self):
        Rs = array(self.Rs)
        i = arange(self.m_modes())
        diags = Rs[:,i,i]
        return log(abs(diags))
