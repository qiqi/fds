import os
import argparse
from subprocess import *
from multiprocessing.pool import ThreadPool

from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

# ---------------------------------------------------------------------------- #

def tangent_initial_condition(degrees_of_freedom, subspace_dimension):
    random.seed(12)
    W = random.rand(degrees_of_freedom, subspace_dimension)
    W, _ = linalg.qr(W)
    w = zeros(degrees_of_freedom)
    return W, w

class TimeDilation:
    def __init__(self, solve, u0, parameter, run_id, time_per_step=1):
        dof = u0.size
        u0p, _ = solve(u0, parameter, 1, run_id)
        self.dxdt = (u0p - u0) / time_per_step
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
    win = sin(linspace(0, pi, a.shape[0]))**2
    return (a * win[:,newaxis]).sum(0) / win.sum()

# -------------------------------- main loop --------------------------------- #

def finite_difference_shadowing(
        solve, u0, parameter, subspace_dimension, num_segments,
        steps_per_segment, runup_steps, epsilon=1E-6, verbose=0):

    degrees_of_freedom = u0.size

    J_hist = []
    G_lss = []
    g_lss = []
    G_dil = []
    g_dil = []
    grad_hist = []

    u0, J0 = solve(u0, parameter, runup_steps, 'runup')
    time_dil = TimeDilation(solve, u0, parameter, 'time_dilation_initial')

    V, v = tangent_initial_condition(degrees_of_freedom, subspace_dimension)
    lss = LssTangent()
    for i in range(num_segments):
        V = time_dil.project(V)
        v = time_dil.project(v)

        threads = ThreadPool()
        run_id = 'segment{0:02d}_baseline'.format(i)
        res_0 = threads.apply_async(
                solve, (u0, parameter, steps_per_segment, run_id))
        # solve homogeneous tangents
        res_h = []
        for j in range(subspace_dimension):
            u1 = u0 + V[:,j] * epsilon
            run_id = 'segment{0:02d}_init_perturb{1:03d}'.format(i, j)
            res_h.append(threads.apply_async(
                solve, (u1, parameter, steps_per_segment, run_id)))
        # solve inhomogeneous tangent
        u1 = u0 + v * epsilon
        run_id = 'segment{0:02d}_param_perturb'.format(i)
        res_i = threads.apply_async(
                solve, (u1, parameter + epsilon, steps_per_segment, run_id))

        u0p, J0 = res_0.get()
        J_hist.append(J0)
        # get homogeneous tangents
        G = []
        for j in range(subspace_dimension):
            u1p, J1 = res_h[j].get()
            V[:,j] = (u1p - u0p) / epsilon
            G.append((J1.mean(0) - J0.mean(0)) / epsilon)
        # get inhomogeneous tangent
        u1p, J1 = res_i.get()
        v, g = (u1p - u0p) / epsilon, (J1.mean(0) - J0.mean(0)) / epsilon

        G_lss.append(G)
        g_lss.append(g)

        # time dilation contribution
        time_dil = TimeDilation(solve, u0p, parameter,
                                'time_dilation_{0:02d}'.format(i))
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))

        lss.checkpoint(V, v)

        alpha = lss.solve()
        grad_lss = (alpha[:,:,newaxis] * array(G_lss)).sum(1) + array(g_lss)
        J = array(J_hist)
        dJ = J.mean((0,1)) - J[:,-1]
        dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
        grad_dil = dil[:,newaxis] * dJ

        grad_hist.append(windowed_mean(grad_lss) + windowed_mean(grad_dil))

        if verbose:
            print('LSS gradient = ', grad_hist[-1])
        if isinstance(verbose, str):
            savez('lss.npz',
                  G_lss=G_lss,
                  g_lss=g_lss,
                  G_dil=G_dil,
                  g_dil=g_dil,
                  R=lss.Rs,
                  b=lss.bs,
                  grad_lss=grad_lss,
                  grad_dil=grad_dil,
                  grad_hist=grad_hist
            )

        # replace initial condition
        u0 = u0p

    return J.mean((0,1)), mean(grad_lss, 0) + mean(grad_dil, 0)
