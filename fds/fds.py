import os
import pickle
import argparse
from multiprocessing import Manager

from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

from .segment import run_segment

# ---------------------------------------------------------------------------- #

def tangent_initial_condition(degrees_of_freedom, subspace_dimension):
    random.seed(12)
    W = random.rand(degrees_of_freedom, subspace_dimension)
    W, _ = linalg.qr(W)
    w = zeros(degrees_of_freedom)
    return W, w

class TimeDilation:
    def __init__(self, run, u0, parameter, run_id, interprocess):
        dof = u0.size
        u0p, _ = run(u0, parameter, 1, run_id, interprocess)
        self.dxdt = (u0p - u0)   # time step size turns out cancelling out
        self.dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)

    def contribution(self, v):
        return dot(self.dxdt, v) / (self.dxdt**2).sum()

    def project(self, v):
        dv = outer(self.dxdt_normalized, dot(self.dxdt_normalized, v))
        return v - dv.reshape(v.shape)

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

def windowed_mean(a):
    win = sin(linspace(0, pi, a.shape[0]+2)[1:-1])**2
    return (a * win[:,newaxis]).sum(0) / win.sum()

def lss_gradient(lss, G_lss, g_lss, J, G_dil, g_dil):
    alpha = lss.solve()
    grad_lss = (alpha[:,:,newaxis] * array(G_lss)).sum(1) + array(g_lss)
    J = array(J)
    dJ = J.mean((0,1)) - J[:,-1]
    steps_per_segment = J.shape[1]
    dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
    grad_dil = dil[:,newaxis] * dJ
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)

def checkpoint(checkpoint_file, V, v, lss, G_lss, g_lss, J, G_dil, g_dil):
    print(lss_gradient(lss, G_lss, g_lss, J, G_dil, g_dil))
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'V': V, 'v': v,
            'lss': lss, 'G_lss': G_lss, 'g_lss': g_lss,
            'J': J, 'G_dil': G_dil, 'g_dil': g_dil
        }, f)

def continue_finite_difference_shadowing(
        run, u0, parameter, V, v, lss,
        G_lss, g_lss, J_hist, G_dil, g_dil,
        num_segments, steps_per_segment, epsilon=1E-6,
        checkpoint_path=None, simultaneous_runs=None):
    """
    """
    assert lss.m_segments() == len(G_lss) \
                            == len(g_lss) \
                            == len(J_hist) \
                            == len(G_dil) \
                            == len(g_dil)
    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    time_dil = TimeDilation(
            run, u0, parameter, 'time_dilation_initial', interprocess)

    for i in range(lss.m_segments(), num_segments):
        V = time_dil.project(V)
        v = time_dil.project(v)

        u0, V, v, J0, G, g = run_segment(
                run, u0, V, v, parameter, i, steps_per_segment,
                epsilon, simultaneous_runs, interprocess)
        J_hist.append(J0)
        G_lss.append(G)
        g_lss.append(g)

        # time dilation contribution
        time_dil = TimeDilation(run, u0, parameter,
                                'time_dilation_{0:02d}'.format(i), interprocess)
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))
        lss.checkpoint(V, v)

        if checkpoint_path:
            filename = 'm{0}_segment{1}'.format(lss.K_modes, lss.m_segments)
            checkpoint(os.path.join(checkpoint_path, filename),
                       V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil)
    return u0, V, v

def finite_difference_shadowing(
        run, u0, parameter, subspace_dimension, num_segments,
        steps_per_segment, runup_steps, epsilon=1E-6,
        checkpoint_path=None, simultaneous_runs=None):
    '''
    run: a function in the form
         u1, J = run(u0, parameter, steps, run_id, interprocess)

         inputs  - u0:           init solution, a flat numpy array of doubles.
                   parameter:    design parameter, a single number.
                   steps:        number of time steps, an int.
                   run_id:       a unique identifier, a string,
                                 e.g., "segment02_init_perturb003".
                   interprocess: a tuple of (lock, dict) for
                                 synchronizing between different runs.
                                 lock: a multiprocessing.Manager.Lock object.
                                 dict: a multiprocessing.Manager.dict object.
         outputs - u1:           final solution, a flat numpy array of doubles,
                                 must be of the same size as u0.
                   J:            quantities of interest, a numpy array of shape
                                 (steps, n_qoi), where n_qoi is an arbitrary
                                 but consistent number, # quantities of interest.
    '''
    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    if runup_steps > 0:
        u0, _ = run(u0, parameter, runup_steps, 'runup', interprocess)

    V, v = tangent_initial_condition(u0.size, subspace_dimension)
    lss = LssTangent()
    G_lss, g_lss = [], []
    J_hist = []
    G_dil, g_dil = [], []

    continue_finite_difference_shadowing(
            run, u0, parameter,
            V, v,
            lss, G_lss, g_lss,
            J_hist, G_dil, g_dil,
            num_segments, steps_per_segment, epsilon,
            checkpoint_path, simultaneous_runs)
    G = lss_gradient(lss, G_lss, g_lss, J_hist, G_dil, g_dil)
    return array(J_hist).mean((0,1)), G
