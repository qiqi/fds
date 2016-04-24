import os
import struct
from subprocess import *

import f90nml
from pylab import *
from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

# ====================== time horizon parameters ============================= #
## time horizon start
m0 = 16500
## time segment size
dm = 200
## number of time segments
K = 60
## total time
dt = 0.001
dT = dm * dt
# number of homogeneous adjoints
p = 2
# ====================== time horizon parameters ============================= #

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

n = int(open('solver/n').read())

# compute check points
t_chkpts = m0 + dm * arange(K+1)
J_hist = zeros([K,dm])

s0, eps = 0, 1E-7
w0, J0 = solver(ones(n), s0, m0)

# Set Tangent Terminal Conditions
random.seed(12)
W = random.rand(n,p)
[QT,RT] = linalg.qr(W)
W = QT

# Loop over all time segments, solve p homogeneous and 1 inhomogeneous
# adjoint on each

Rs = zeros([K,p,p])
bs = zeros([K,p])
gs = zeros([K,p])
h = zeros(K)
gs2 = zeros([K,p])
h2 = zeros(K)

wh = zeros(n)

w1, _ = solver(w0, s0, 1)
dxdt = (w1 - w0) / dt

for i in range(K):
    t_strt = t_chkpts[i]
    t_end = t_chkpts[i+1]
    # print "segment {0}: [{1},{2}]".format(str(i), t_strt, t_end)
    # print(w0)

    dxdt_normalized = dxdt / linalg.norm(dxdt)
    P = eye(n) - outer(dxdt_normalized, dxdt_normalized)
    W = dot(P, W)
    wh = dot(P, wh)

    # solve homogeneous tangents
    w0p, J0 = solver(w0, s0, t_end - t_strt)
    J_hist[i] = J0
    for j in range(p):
        w1p, J1 = solver(w0 + W[:,j] * eps, s0, t_end - t_strt)
        W[:,j], gs[i,j] = (w1p - w0p) / eps, (J1.mean() - J0.mean()) / eps

    # solve inhomogeneous tangent
    w1p, J1 = solver(w0 + wh * eps, s0 + eps, t_end - t_strt)
    wh, h[i] = (w1p - w0p) / eps, (J1.mean() - J0.mean()) / eps

    w0 = w0p

    # w1, _ = solver(w0, s0, 1)
    # dxdt = (w1 - w0) / dt
    # dxdt_normalized = dxdt / linalg.norm(dxdt)
    # P = eye(n) - outer(dxdt_normalized, dxdt_normalized)
    # W = dot(P, W)
    # wh = dot(P, wh)

    w1, _ = solver(w0, s0, 1)
    dxdt = (w1 - w0) / dt
    gs2[i,:] = dot(dxdt, W) / (dxdt*dxdt).sum()
    h2[i] = dot(dxdt, wh) / (dxdt*dxdt).sum()

    # QR decomposition
    [Q,R] = linalg.qr(W)
    Rs[i,:,:] = R
    bs[i,:] = dot(Q.T, wh)

    # Set terminal conditions for next segment
    W = Q
    wh = wh - dot(Q, bs[i,:])

    # print R, bs[i,:], wh

J_mean = J_hist.mean()

# form KKT system and solve
d = ones(K)
eyes = eye(p,p) / d[:,newaxis,newaxis]
B = -sparse.bsr_matrix((eyes, r_[1:K+1], r_[:K+1]))\
          + sparse.bsr_matrix((Rs, r_[:K],r_[:K+1]),\
          shape=(K*p, (K+1)*p))
B = B.tocsr()
A = B * B.T
rhs = ravel(bs)

alpha = -(B.T * splinalg.spsolve(A, rhs)).reshape([K+1,-1])[:-1]
grad0 = (alpha * gs).sum(1) + h
dJ = J_hist.mean() - J_hist[:,-1]
grad1 = ((alpha * gs2).sum(1) + h2) / dT * dJ
print(grad0.mean(), grad1.mean(), grad0.mean() + grad1.mean())
