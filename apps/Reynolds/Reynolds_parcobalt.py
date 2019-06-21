import h5py
import os
import sys
import time
import shutil
import tempfile
import argparse
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *
from fds.checkpoint import *
from fds.cobalt import CobaltManager

ALPHA = 0.0                 # nominal angle of attach
XMACH = 0.3                 # nominal xmach parameter
M_MODES = 6 #16             # number of Lyapunov exponents to compute
K_SEGMENTS = 500            # number of time segments
STEPS_PER_SEGMENT = 20      # number of time-steps per time-segment
STEPS_RUNUP = 0             # number of time-steps for run up time
SLEEP_SECONDS_FOR_IO = 5    # how long to wait for file IO to sync
MPI_LES = 1024               # Number of MPI ranks per LES simulation
MPI_PER_NODE = 16           # Number of MPI ranks per node
SIMULTANEOUS_RUNS = M_MODES + 1 + 1      # Total number of LES run on parallel (baseline + initial condition perturbations + parameter perturbations)
NODES_PER_RUN = MPI_LES/MPI_PER_NODE
#partShape = '4x4x4x4x2'
jobShape = '2x2x2x2x1'
MPI_FDS = MPI_LES

REF_WORK_PATH = os.path.join(os.sep,'projects','LESOpt','pablof','fds','problem','cylinderRe500_3d')

parser = argparse.ArgumentParser()
parser.add_argument('--xmach', action='store_true')
parser.add_argument('--alpha', action='store_true')
args = parser.parse_args()

if not (args.xmach or args.alpha):
    sys.stderr.write('Must specify --xmach or --alpha\n')
    sys.exit(-1)

if args.xmach:
    BASE_PATH = os.path.join(my_path, 'Reynolds_xmach')
    S_BASELINE = XMACH
elif args.alpha:
    BASE_PATH = os.path.join(my_path, 'Reynolds_alpha')
    S_BASELINE = ALPHA
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

Reynolds_bin = os.path.join(os.sep,'projects','LESOpt','pablof','fds','Reynolds','Reynolds')

Cobalt = CobaltManager(jobShape, SIMULTANEOUS_RUNS) # TODO: We assume block=partition. This seems to be required is blocks are not big enough
print('No. available blocks = ', len(Cobalt.blocks))

def get_host_dir(run_id):
    return os.path.join(BASE_PATH, run_id)

def spawn_compute_job(exe, args, **kwargs):
    global Cobalt
    if 'interprocess' in kwargs:
        Cobalt.interprocess = kwargs['interprocess']
        del kwargs['interprocess']
    corner = Cobalt.get_corner()
    returncode = call(['runjob', '-n', str(MPI_FDS), 
                       '-p', str(MPI_PER_NODE),
                       '--block', Cobalt.partition,
                       '--corner', corner,
                       '--shape', jobShape,
                       '--exp-env', 'PYTHONPATH',
                       '--verbose', 'INFO',
                       ':', exe] + args, **kwargs)
    Cobalt.free_corner(corner)
    return returncode

def lift_drag_pressure_from_text(text, xmach):
    lift_drag_pressure = []
    for line in text.split('\n'):
        line = line.strip().split()
        if len(line) == 6 and line[0] == 'Lift' and line[2] == 'Drag' and line[4] == 'BackPressure':
            cl, cd, backPressure = float(line[1]), float(line[3]), float(line[5])
            q = 0.5 * xmach**2   # assuming density 1, is that right?
            lift_drag_pressure.append([cl, cd, q * cl, q * cd, backPressure])
    return array(lift_drag_pressure)

def solve(hdf5_inputFile_path, s, nsteps, run_id, interprocess):
    Cobalt.interprocess = interprocess
    if args.xmach:
        xmach, alpha = s, ALPHA
    elif args.alpha:
        xmach, alpha = XMACH, s
    print('Starting solve, xmach, alpha, nsteps, run_id = ', xmach, alpha, nsteps, run_id)

    # Get name of relevant files:
    work_path = get_host_dir(run_id)
    HDF5_outpuFile_path = os.path.join(work_path, 'output.h5')
    initial_data_files = [os.path.join(work_path, 'initialData' + str(i) + '.bin')
                          for i in range(MPI_LES)]
    final_data_files = [os.path.join(work_path, 'finalData' + str(i) + '.bin')
                        for i in range(MPI_LES)]
    dataStruct_files = [os.path.join(work_path, 'dataStructs' + str(i) + '.bin')
                        for i in range(MPI_LES)]
    lift_drag_pressure_file = os.path.join(work_path, 'lift_drag_pressure.txt')
    lift_drag_pressure_file_tmp = os.path.join(work_path, 'lift_drag_pressure_tmp.txt')

    blockName = Cobalt.partition        # TODO: We assume block = partition here
    while not all([os.path.exists(f) for f in final_data_files]) or \
            not os.path.exists(lift_drag_pressure_file_tmp):
        if os.path.exists(work_path):
            shutil.rmtree(work_path)
        os.mkdir(work_path)
        outfile = os.path.join(work_path, 'output')
        with open(outfile, 'w', 8) as f:
            # Get corner:
            while True:
                corner = Cobalt.get_corner()
                if corner != -1:
                    break

            # Generate input files
            Popen(['runjob','--block', str(blockName),
                   '--corner', str(corner), '--shape', str(jobShape),
                   '-n', str(MPI_LES),'-p', str(MPI_PER_NODE),':',
                   sys.executable, 'from_hdf5_to_ReynoldsInputFiles', hdf5_inputFile_path, work_path,
                   REF_WORK_PATH],cwd=work_path, stdout=f, stderr=f).wait()
            time.sleep(SLEEP_SECONDS_FOR_IO)

            # Run MPI job:
            Popen(['runjob','--block', str(blockName),
                   '--corner', str(corner), '--shape', str(jobShape),
                   '-n', str(MPI_LES),'-p', str(MPI_PER_NODE),':',
                   Reynolds_bin, os.path.join(work_path, 'dataStructs'),os.path.join(work_path, 'initialData'),
                   os.path.join(work_path, 'finalData'), lift_drag_pressure_file_tmp, str(nsteps), str(xmach), str(alpha)],
                  cwd=work_path, stdout=f, stderr=f).wait()
            time.sleep(SLEEP_SECONDS_FOR_IO)

            # Generate HDF5 file with final solution
            Popen(['runjob','--block', str(blockName),
                       '--corner', str(corner), '--shape', str(jobShape),
                       '-n', str(MPI_LES),'-p', str(MPI_PER_NODE),':',
                       sys.executable, 'from_ReynoldsOutputFiles_to_hdf5', HDF5_outpuFile_path, work_path,
                       REF_WORK_PATH],cwd=work_path, stdout=f, stderr=f).wait()
            time.sleep(SLEEP_SECONDS_FOR_IO)

            # Free corner:
            Cobalt.free_corner(corner)

        if not os.path.exists(lift_drag_pressure_file_tmp) or \
                not all([os.path.exists(f) for f in final_data_files]):
            failed_path = work_path + '_failed'
            if os.path.exists(failed_path):
                shutil.rmtree(failed_path)
            shutil.move(work_path, failed_path)

    lift_drag_pressure = lift_drag_pressure_from_text(open(lift_drag_pressure_file_tmp).read(), xmach)
    os.remove(lift_drag_pressure_file_tmp)
    savetxt(lift_drag_pressure_file, lift_drag_pressure)
    J = loadtxt(lift_drag_pressure_file).reshape([-1,5])

    # Remove files that are no longer required to reduce disk requirements
    for f in final_data_files:
        os.remove(f)
    for f in dataStruct_files:
        os.remove(f)

    assert len(J) == nsteps
    Cobalt.interprocess = None
    return HDF5_outpuFile_path, J

if __name__ == '__main__':
    hdf5_path = os.path.join(BASE_PATH, 'u0.h5')

    # Boot blocks:
    Cobalt.boot_blocks()

    checkpoint = load_last_checkpoint(BASE_PATH, M_MODES)
    if checkpoint is None:
        J, G = shadowing(solve,
                    hdf5_path,
                    S_BASELINE,
                    M_MODES,
                    K_SEGMENTS,
                    STEPS_PER_SEGMENT,
                    STEPS_RUNUP,
                    epsilon=1E-4,
                    checkpoint_path=BASE_PATH,
                    simultaneous_runs=SIMULTANEOUS_RUNS,
                    get_host_dir=get_host_dir,
                    spawn_compute_job=spawn_compute_job)
        #run_ddt=0,
    else:
        J, G = continue_shadowing(solve,
                                  S_BASELINE,
                                  checkpoint,
                                  K_SEGMENTS,
                                  STEPS_PER_SEGMENT,
                                  epsilon=1E-4,
                                  checkpoint_path=BASE_PATH,
                                  simultaneous_runs=SIMULTANEOUS_RUNS,
                                  get_host_dir=get_host_dir,
                                  spawn_compute_job=spawn_compute_job)
        #run_ddt=0,

    # Free blocks:
    Cobalt.free_blocks()
