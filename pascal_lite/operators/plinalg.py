import numpy
from numpy import *
from mpi4py import MPI

def pQR(comm, A):
    # Compute QR factorization A = Q*R

    root = 0
    rank = comm.Get_rank()
    size = comm.Get_size()

# Compute Q1 and R1:
    Q1, R1 = linalg.qr(A)
    Rshape = R1.shape

    if len(Rshape) == 1:
        mR = Rshape[0]
        nR = 1
    elif len(Rshape) == 2:
        mR = Rshape[0]
        nR = Rshape[1]
    else:
        print("R has invalid shape.")

    if mR != nR:
        print("R is not a square matrix.")

    # Gather R1 in root processor:
    sendbuf = R1
    if rank == root:
        recvbuf = numpy.empty((size, mR, nR), dtype='d')
    else:
        recvbuf = None
    comm.Gather(sendbuf, recvbuf, root)

    # Reshape recvbuf and compute Q2:
    if rank == root:
        R1full = recvbuf.reshape(size*mR, nR)

        Q2, R = linalg.qr(R1full)
        sendbuf = Q2
    else:
        R = None
        sendbuf = None

    # Broadcast R
    R = comm.Bcast(R, root)

    # Scatter Q2
    recvbuf = numpy.empty((mR, nR), dtype='d')
    comm.Scatter(sendbuf, recvbuf, root)
    Q2 = recvbuf

    # Compute Q
    Q = linalg.dot(Q1,Q2)

    return Q, R


def pdot(comm, A, B):
    # Compute matrix-matrix product C = A*B.
    # The input data is distributed along the contraction dimensions

    Ashape = A.shape
    Bshape = B.shape

    if len(Ashape) == 1:
        m1 = 1
        n1 = Ashape[0]
    elif len(Ashape) == 2:
        m1 = Ashape[0]
        n1 = Ashape[1]
    else:
        print("A has invalid shape.")

    if len(Bshape) == 1:
        m2 = Bshape[0]
        n2 = 1
    elif len(Bshape) == 2:
        m2 = Bshape[0]
        n2 = Bshape[1]
    else:
        print("B has invalid shape.")

    C_local = dot(A,B)
    C_global = numpy.zeros(m1,n2)

    comm.Allreduce(C_local, C_global, MPI.SUM)

    return C_global

