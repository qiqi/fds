from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

from .timeseries import windowed_mean
from .states import PrimalState, TangentState

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
        Q, R = linalg.qr(V.T)
        b = dot(Q.T, v)
        self.Rs.append(R)
        self.bs.append(b)
        V[:] = Q.T
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
        Schur = B * B.T #+ 1E-5 * sparse.eye(B.shape[0])
        alpha = -(B.T * splinalg.spsolve(Schur, ravel(b)))
        # alpha1 = splinalg.lsqr(B, ravel(bs), iter_lim=10000)
        return alpha.reshape([nseg+1,-1])[:-1]

    def solve_ivp(self):
        a = [zeros(self.bs[0].shape)]
        for i in range(len(self.bs)):
            a.append(dot(self.Rs[i], a[-1]) + self.bs[i])
        return array(a)[:-1]

    def lyapunov_exponents(self):
        R = array(self.Rs)
        i = arange(self.m_modes())
        diags = R[:,i,i]
        return log(abs(diags))

    def lyapunov_covariant_vectors(self):
        exponents = self.lyapunov_exponents().mean(0)
        multiplier = exp(exponents)
        vi = eye(self.m_modes())
        v = [vi]
        for Ri in reversed(self.Rs):
            vi = linalg.solve(Ri, vi) * multiplier
            v.insert(0, vi)
        v = array(v)
        return rollaxis(v, 2)

    def lyapunov_covariant_magnitude_and_sin_angle(self):
        v = self.lyapunov_covariant_vectors()
        v_magnitude = sqrt((v**2).sum(2))
        vv = (v[:,newaxis] * v[newaxis,:]).sum(3)
        cos_angle = (vv / v_magnitude).transpose([1,0,2]) / v_magnitude
        i = arange(cos_angle.shape[0])
        cos_angle[i,i,:] = 1
        sin_angle = sqrt(1 - cos_angle**2)
        return v_magnitude, sin_angle


