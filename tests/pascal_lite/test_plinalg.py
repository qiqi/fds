import sys
import numpy as np
import os


if __name__ == '__main__':
    from pascal_lite.operators.plinalg import pdot, pQR
    try:
        from mpi4py import MPI
    except ImportError:
        pass

    comm = MPI.COMM_WORLD
    rank = comm.rank
    nprocs = comm.Get_size()

    # pQR unit test:
    import os
    import sys
    A = np.loadtxt(sys.argv[1])
    size = A.shape[1]
    start = rank * size/nprocs
    end = (rank + 1) * size/nprocs
    A = A[:,start:end]
    Q, R = pQR(comm,A.T)
    Q = comm.gather(Q, root=0)
    R = comm.gather(R, root=0)

    b = np.loadtxt(sys.argv[2])
    b = b[start:end]
    dot = pdot(comm, A, b)
    dot = comm.gather(dot, root=0)

    if rank == 0:
        Q = np.vstack(Q)
        for i in range(0, nprocs):
            assert np.allclose(dot[0], dot[i])
            assert np.allclose(R[0], R[i])
        np.savetxt(os.path.join(os.path.dirname(sys.argv[1]), 'plinalg_Q.txt'), Q)
        np.savetxt(os.path.join(os.path.dirname(sys.argv[1]), 'plinalg_R.txt'), R[0])
        np.savetxt(os.path.join(os.path.dirname(sys.argv[1]), 'plinalg_dot.txt'), dot[0])

    # pdot unit test:
    #A = np.random.rand(5,1)
    #B = np.random.rand(5,1)
    #C = pdot(comm,A,B)
