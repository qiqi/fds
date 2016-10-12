import os
import sys
import gzip
import shutil
import argparse

import h5py
import numpy as np
import mpi4py
from mpi4py import MPI

from foam_data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Foam time to h5py')
    parser.add_argument('ref_path', type=str, help='Reference OpenFOAM case')
    parser.add_argument('hdf5_input', type=str, help='Input hdf5 file name')
    parser.add_argument('out_path', type=str, help='Output OpenFOAM case path')
    parser.add_argument('time', type=str, help='Time')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    ref_path = find_data_path(comm, args.ref_path, args.time)
    data = list(map(DataLoader(ref_path), sorted(os.listdir(ref_path))))
    data = np.hstack(data)

    data_size = np.zeros(comm.size, int)
    data_size[comm.rank] = data.size
    comm.Allreduce(MPI.IN_PLACE, data_size, MPI.SUM)
    i_start = data_size[:comm.rank].sum()
    i_end = data_size[:comm.rank+1].sum()

    handle = h5py.File(args.hdf5_input, 'r', driver='mpio', comm=comm)
    data = handle['/field'][i_start:i_end]
    handle.close()

    out_path = find_data_path(comm, args.out_path, args.time, True)
    writer = DataWriter(ref_path, out_path)
    for fname in sorted(os.listdir(ref_path)):
        if not fname.endswith('.gz'):
            continue
        written_data = writer(data, fname)
        assert written_data.size <= data.size
        data = data[written_data.size:]
    assert data.size == 0

    if comm.rank == 0:
        shutil.copytree(os.path.join(args.ref_path, 'system'),
                        os.path.join(args.out_path, 'system'))
        shutil.copytree(os.path.join(args.ref_path, 'constant'),
                        os.path.join(args.out_path, 'constant'))
    proc_path = 'processor{0}'.format(comm.rank)
    shutil.copytree(os.path.join(args.ref_path, proc_path, 'constant'),
                    os.path.join(args.out_path, proc_path, 'constant'))
