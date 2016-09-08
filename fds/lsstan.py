import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg

from .timeseries import windowed_mean
import pascal_lite as pascal

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
        #Q, R = pascal.qr(V.T)
        #b = pascal.dot(Q.T, v)
        #V[:] = Q.T
        #v -= pascal.dot(Q, b)
        Q, R = pascal.qr_transpose(V)
        b = pascal.dot(Q, v)
        #V[:] = Q
        #v -= pascal.dot(b, Q)
        V = Q
        v = v - pascal.dot(b, Q)

        self.Rs.append(R)
        self.bs.append(b)
        return V, v

    def solve(self):
        R, b = np.array(self.Rs), np.array(self.bs)
        assert R.ndim == 3 and b.ndim == 2
        assert R.shape[0] == b.shape[0]
        assert R.shape[1] == R.shape[2] == b.shape[1]
        nseg, subdim = b.shape
        eyes = np.eye(subdim, subdim) * np.ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, np.r_[1:nseg+1], np.r_[:nseg+1]))
        D = sparse.bsr_matrix((R, np.r_[:nseg], np.r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        Schur = B * B.T #+ 1E-5 * sparse.eye(B.shape[0])
        alpha = -(B.T * splinalg.spsolve(Schur, np.ravel(b)))
        # alpha1 = splinalg.lsqr(B, ravel(bs), iter_lim=10000)
        return alpha.reshape([nseg+1,-1])[:-1]

    def solve_ivp(self):
        a = [np.zeros(self.bs[0].shape)]
        for i in range(len(self.bs)):
            a.append(np.dot(self.Rs[i], a[-1]) + self.bs[i])
        return array(a)[:-1]

    def lyapunov_exponents(self):
        R = np.array(self.Rs)
        i = np.arange(self.m_modes())
        diags = R[:,i,i]
        return np.log(abs(diags))

    def lyapunov_covariant_vectors(self):
        exponents = self.lyapunov_exponents().mean(0)
        multiplier = np.exp(exponents)
        vi = np.eye(self.m_modes())
        v = [vi]
        for Ri in reversed(self.Rs):
            vi = np.linalg.solve(Ri, vi) * multiplier
            v.insert(0, vi)
        v = np.array(v)
        return np.rollaxis(v, 2)

    def lyapunov_covariant_magnitude_and_sin_angle(self):
        v = self.lyapunov_covariant_vectors()
        v_magnitude = np.sqrt((v**2).sum(2))
        vv = (v[:,np.newaxis] * v[np.newaxis,:]).sum(3)
        cos_angle = (vv / v_magnitude).transpose([1,0,2]) / v_magnitude
        i = np.arange(cos_angle.shape[0])
        cos_angle[i,i,:] = 1
        sin_angle = np.sqrt(1 - cos_angle**2)
        return v_magnitude, sin_angle


