import numpy
from numpy import *
try:
    from mpi4py import MPI
except ImportError:
    pass

def pQR(comm, A):
    # Compute QR factorization A = Q*R. Q is assumed to be a tall matrix whose rows are distributed among processors.

    Ashape = A.shape
    assert Ashape[0] >= Ashape[1]

    root = 0
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Compute Q1 and R1:
    Q1, R1 = linalg.qr(A)
    Rshape = R1.shape

    assert len(Rshape) == 1 or len(Rshape) == 2
    if len(Rshape) == 1:
        mR = Rshape[0]
        nR = 1
    elif len(Rshape) == 2:
        mR = Rshape[0]
        nR = Rshape[1]

    assert mR == nR

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
        R = numpy.empty((mR,nR), dtype='d')
        sendbuf = None

    # Broadcast R
    comm.Bcast(R, root)

    # Scatter Q2
    recvbuf = numpy.empty((mR, nR), dtype='d')
    comm.Scatter(sendbuf, recvbuf, root)
    Q2 = recvbuf

    # Compute Q
    Q = dot(Q1,Q2)

    return Q, R


def pdot(comm, A, B):
    # Compute matrix-matrix product C = A*B.
    # The input data is distributed along the contraction dimensions

    Ashape = A.shape
    Bshape = B.shape

    assert len(Ashape) == 1 or len(Ashape) == 2
    if len(Ashape) == 1:
        m1 = 1
        n1 = Ashape[0]
    elif len(Ashape) == 2:
        m1 = Ashape[0]
        n1 = Ashape[1]

    assert len(Bshape) == 1 or len(Bshape) == 2
    if len(Bshape) == 1:
        m2 = Bshape[0]
        n2 = 1
    elif len(Bshape) == 2:
        m2 = Bshape[0]
        n2 = Bshape[1]

    #C_local = dot(A,B)
    #C_global = numpy.zeros((m1,n2))
    C_local = (A*B).sum(-1)
    C_global = numpy.zeros_like(C_local)

    comm.Allreduce(C_local, C_global, MPI.SUM)

    return C_global


#comm = MPI.COMM_WORLD

## pQR unit test:
#A = numpy.random.rand(10,3)
#print comm.Get_rank, A
#Q, R = pQR(comm,A)
#print comm.Get_rank, Q
#print comm.Get_rank, R

## pdot unit test:
#A = numpy.random.rand(5,1)
#print(comm.Get_rank)
#print(A)
#B = numpy.random.rand(5,1)
#print(comm.Get_rank)
#print(B)
#C = pdot(comm,A,B)
#print(C)
