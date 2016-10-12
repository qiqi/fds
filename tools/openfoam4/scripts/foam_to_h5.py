import os
import sys
import argparse

import h5py
import numpy as np
from mpi4py import MPI

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(my_path)

from foam_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Foam time to h5py')
    parser.add_argument('foam_path', type=str, help='Path to OpenFOAM case')
    parser.add_argument('time', type=str, help='Time to convert')
    parser.add_argument('output', type=str, help='hdf5 output file')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    data_path = find_data_path(comm, args.foam_path, args.time)
    data = list(map(DataLoader(data_path), sorted(os.listdir(data_path))))
    data = np.hstack(data)

    data_size = np.zeros(comm.size, int)
    data_size[comm.rank] = data.size
    comm.Allreduce(MPI.IN_PLACE, data_size, MPI.SUM)
    i_start = data_size[:comm.rank].sum()
    i_end = data_size[:comm.rank+1].sum()

    handle = h5py.File(args.output, 'w', driver='mpio', comm=comm)
    d = handle.create_dataset('field', shape=(data_size.sum(),), dtype='d')
    d[i_start:i_end] = data
    handle.close()
