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
        R, b = array(self.Rs), array(self.bs)
        assert R.ndim == 3 and b.ndim == 2
        assert R.shape[0] == b.shape[0]
        assert R.shape[1] == R.shape[2] == b.shape[1]
        nseg, subdim = b.shape
        eyes = eye(subdim, subdim) * ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, r_[1:nseg+1], r_[:nseg+1]))
        D = sparse.bsr_matrix((R, r_[:nseg], r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        alpha = -(B.T * splinalg.spsolve(B * B.T, ravel(b)))
        # alpha1 = splinalg.lsqr(B, ravel(bs), iter_lim=10000)
        return alpha.reshape([nseg+1,-1])[:-1]

    def lyapunov_exponents(self):
        R = array(self.Rs)
        i = arange(self.m_modes())
        diags = R[:,i,i]
        return log(abs(diags))

    def lyapunov_covariant_vectors(self):
        ai = eye(self.m_modes())
        a = [ai]
        for Ri in reversed(self.Rs):
            ai = linalg.solve(Ri, ai)
            a.insert(0, ai)
        a = array(a)
        a /= abs(a).mean((0,1))
        return rollaxis(a, 2)
