import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg

from .state import qr_transpose_states, state_dot

class LssTangent:
    def __init__(self, m_modes):
        self.m_modes = m_modes
        self.Rs = []
        self.bs = []

    def K_segments(self):
        assert len(self.Rs) == len(self.bs)
        return len(self.Rs)

    def checkpoint(self, V, v):
        assert self.m_modes == len(V)
        Q, R = qr_transpose_states(V)
        b = state_dot(Q, v)
        V = Q
        v = v - state_dot(b, Q)

        self.Rs.append(R)
        self.bs.append(b)
        return V, v

    def adjoint_checkpoint(self, V, w, b_adj):
        Q, R = qr_transpose_states(V)
        c = state_dot(Q, w) - b_adj
        w = w - state_dot(c, Q)
        return w

    def solve(self):
        if len(self.bs) == 1:
            return np.zeros_like(self.bs)
        R, b = np.array(self.Rs[1:]), np.array(self.bs[1:])
        assert R.ndim == 3 and b.ndim == 2
        assert R.shape[0] == b.shape[0]
        assert R.shape[1] == R.shape[2] == b.shape[1]
        nseg, subdim = b.shape
        eyes = np.eye(subdim, subdim) * np.ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, np.r_[1:nseg+1], np.r_[:nseg+1]))
        D = sparse.bsr_matrix((R, np.r_[:nseg], np.r_[:nseg+1]),
                              shape=matrix_shape)
        B = (D - I).tocsr()
        Schur = B * B.T #+ 1E-5 * sparse.eye(B.shape[0])
        alpha = -(B.T * splinalg.spsolve(Schur, np.ravel(b)))
        # alpha1 = splinalg.lsqr(B, ravel(bs), iter_lim=10000)
        return alpha.reshape([nseg+1,-1])

    def adjoint(self, alpha_adj):
        R = np.array(self.Rs[1:])
        assert R.ndim == 3
        assert R.shape[0] == alpha_adj.shape[0] - 1
        assert R.shape[1] == R.shape[2] == alpha_adj.shape[1]
        nseg, subdim = R.shape[:2]
        eyes = np.eye(subdim, subdim) * np.ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, np.r_[1:nseg+1], np.r_[:nseg+1]))
        D = sparse.bsr_matrix((R, np.r_[:nseg], np.r_[:nseg+1]),
                              shape=matrix_shape)
        B = (D - I).tocsr()
        Schur = B * B.T #+ 1E-5 * sparse.eye(B.shape[0])
        b_adj = -splinalg.spsolve(Schur, B * np.ravel(alpha_adj))
        return np.vstack([np.zeros([1, subdim]),
                          b_adj.reshape([nseg, subdim])])

    def solve_ivp(self):
        a = [np.zeros(self.bs[0].shape)]
        for i in range(1, len(self.bs)):
            a.append(np.dot(self.Rs[i], a[-1]) + self.bs[i])
        return array(a)

    def lyapunov_exponents(self, segment_range=None):
        R = np.array(self.Rs[1:])
        if segment_range is not None:
            R = R[slice(*segment_range)]
        i = np.arange(self.m_modes)
        diags = R[:,i,i]
        return np.log(abs(diags))

    def lyapunov_covariant_vectors(self):
        exponents = self.lyapunov_exponents().mean(0)
        multiplier = np.exp(exponents)
        vi = np.eye(self.m_modes)
        v = [vi]
        for Ri in reversed(self.Rs[1:]):
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


