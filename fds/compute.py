import os
import sys
import numpy as np
import cPickle as pkl
import subprocess

import pascal_lite as pascal

try:    
    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    import h5py
except ImportError:
    pass

def run_compute(outputs):
    graph = pascal.ComputationalGraph([x.value for x in outputs])
    sample_input = [x for x in graph.input_values if x.field is not None][0]
    if isinstance(sample_input.field, np.ndarray):
        serial_compute(sample_input, outputs, graph)
    else:
        mpi_compute(sample_input, outputs, graph)
    return

def get_inputs(x, size):
    if x is pascal.builtin.ZERO:
        return np.zeros(size)
    elif x in pascal.builtin.RANDOM:
        shape = x.shape + (size,)
        return np.random.rand(*shape)
    elif isinstance(x.field, np.ndarray):
        return x.field
    else:
        return mpi_read_field(x.field)

def serial_compute(sample_input, outputs, graph):
    size = sample_input.field.shape[0]
    print len(graph.input_values)
    inputs = lambda x: get_inputs(x, size)
    actual_outputs = graph(inputs)
    for index, output in enumerate(outputs):
        output.value.field = actual_outputs[index]
    return 

def mpi_compute(*mpi_inputs):#, spawn_job=None):

    pkl_file = os.path.abspath('graph.pkl')
    with open(pkl_file, 'w') as f:
        pkl.dump(mpi_inputs, f)

    # spawn job and wait for result
    worker_file = os.path.join(os.path.abspath(__file__))
    spawn_job=None
    if spawn_job is None:
        subprocess.call(['mpirun', worker_file, pkl_file])
    else:
        spawn_job(worker_file, [pkl_file])
    return 

def mpi_range(size):
        mpi_size = size / mpi.Get_size()
        start = mpi.rank * mpi_size
        end = min(size, start + mpi_size)
        return start, end

def mpi_read_field(field_file):
    handle = h5py.File(field_file, 'r', driver='mpio', comm=mpi)
    field = handle['/field']
    start, end = mpi_range(field.shape[0])
    field = field[start:end].copy()
    #field = np.loadtxt(field_file)
    return field

def mpi_write_field(field, field_file):
    handle = h5py.File(field_file, 'w', driver='mpio', comm=mpi)
    field = handle['/field']
    start, end = mpi_range(field.shape[0])
    field[start:end] = field
    #np.savetxt(field_file, field)
    return

def mpi_compute_worker():
    zero = pascal.builtin.ZERO
    random = pascal.builtin.RANDOM[0]

    pkl_file = sys.argv[1]
    with open(pkl_file) as f:
        sample_input, outputs, graph = pkl.load(f)
    
    # read the inputs for the graph
    size = mpi_read_field(sample_input.field).shape[0]
    inputs = lambda x: get_inputs(x, size)
    
    # perform the computation
    actual_outputs = graph(inputs)

    # write the outputs in the parent directory for the job
    for index, output in enumerate(outputs):
        parent_dir = os.path.dirname(output.field)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        mpi_write_field(actual_outputs[index], output.field)
    return

if __name__ == "__main__":
    mpi_compute_worker()
