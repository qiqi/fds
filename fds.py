import os
import argparse
from subprocess import *

from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

# -------------------------------- parameters -------------------------------- #

parser = argparse.ArgumentParser(description='Finite Difference Shadowing')
parser.add_argument('--runup_steps', type=int, default=5000)
parser.add_argument('--steps_per_segment', type=int, default=1000)
parser.add_argument('--num_segments', type=int, default=5)
parser.add_argument('--time_per_step', type=float, default=0.001)
parser.add_argument('--subspace_dimension', type=int, default=2)
parser.add_argument('--parameter', type=float, default=0.0)
parser.add_argument('--epsilon', type=float, default=1E-6)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #

def solver(u, s, nsteps):
    os.chdir('solver')
    with open('input.bin', 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open('param.bin', 'wb') as f:
        f.write(asarray(s, dtype='>d').tobytes())
    call(["./solver", str(int(nsteps))])
    with open('output.bin', 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open('objective.bin', 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    os.chdir('..')
    return out, J

def tangent_initial_condition(degrees_of_freedom, subspace_dimension):
    random.seed(12)
    W = random.rand(degrees_of_freedom, subspace_dimension)
    W, _ = linalg.qr(W)
    w = zeros(degrees_of_freedom)
    return W, w

class TimeDilation:
    def __init__(self, u0):
        dof = u0.size
        u0p, _ = solver(u0, args.parameter, 1)
        self.dxdt = (u0p - u0) / args.time_per_step
        dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)
        self.P = eye(dof) - outer(dxdt_normalized, dxdt_normalized)

    def contribution(self, v):
        return dot(self.dxdt, v) / (self.dxdt**2).sum()

    def project(self, v):
        return dot(self.P, v)

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

# -------------------------------- main loop --------------------------------- #

u0 = loadtxt('solver/u0')
degrees_of_freedom = u0.size

J_hist = zeros([args.num_segments, args.steps_per_segment])
G_lss = []
g_lss = []
G_dil = []
g_dil = []

u0, J0 = solver(u0, args.parameter, args.runup_steps)
time_dil = TimeDilation(u0)

V, v = tangent_initial_condition(degrees_of_freedom, args.subspace_dimension)
lss = LssTangent()
for i in range(args.num_segments):
    V = time_dil.project(V)
    v = time_dil.project(v)

    u0p, J0 = solver(u0, args.parameter, args.steps_per_segment)
    J_hist[i] = J0

    # solve homogeneous tangents
    G = empty(args.subspace_dimension)
    for j in range(args.subspace_dimension):
        u1 = u0 + V[:,j] * args.epsilon
        u1p, J1 = solver(u1, args.parameter, args.steps_per_segment)
        V[:,j] = (u1p - u0p) / args.epsilon
        G[j] = (J1.mean() - J0.mean()) / args.epsilon

    # solve inhomogeneous tangent
    u1 = u0 + v * args.epsilon
    u1p, J1 = solver(u1, args.parameter + args.epsilon, args.steps_per_segment)
    v, g = (u1p - u0p) / args.epsilon, (J1.mean() - J0.mean()) / args.epsilon

    G_lss.append(G)
    g_lss.append(g)

    # time dilation contribution
    time_dil = TimeDilation(u0p)
    G_dil.append(time_dil.contribution(V))
    g_dil.append(time_dil.contribution(v))

    lss.checkpoint(V, v)

    # replace initial condition
    u0 = u0p

alpha = lss.solve()
grad0 = (alpha * G_lss).sum(1) + g_lss
dJ = J_hist.mean() - J_hist[:,-1]
time_per_segment = args.steps_per_segment * args.time_per_step
grad1 = ((alpha * G_dil).sum(1) + g_dil) / time_per_segment * dJ

print(J_hist.mean())
print(grad0.mean() + grad1.mean())
