import os
import sys
import subprocess
import numpy as np
import shutil
import glob

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
fileName = os.path.abspath(__file__)
python = sys.executable

nProcessors = 16
nProcsPerNode = 8
nRuns = 128
subBlockShape = '1x1x1x1x2'

parameter = 1.0
dims = 20
segments = 20
steps = 10000

time = 2.0
source = '/projects/LESOpt/talnikar/'
problem = 'cylinder.py'
case = source + 'cylinder/fds/'

def getTime(time):
    stime = str(time)
    if time.is_integer():
        stime = str(int(time))
    return stime
stime = getTime(time)

fieldNames = ['rho', 'rhoU', 'rhoE']
program = source + 'local/src/adFVM/apps/problem.py'

reference = [1., 200., 2e5]

def getParallelInfo():
    import h5py
    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    rank = mpi.rank 
    
    with h5py.File(case + 'mesh.hdf5', 'r') as mesh:
        nCount = mesh['parallel/end'][rank]-mesh['parallel/start'][rank]
        nInternalCells = nCount[4]
        nGhostCells = nCount[2]-nCount[3]
        nCells = nInternalCells + nGhostCells
        cellStart = mpi.exscan(nCells)
        if cellStart == None:
            cellStart = 0
        cellEnd = cellStart + nInternalCells

    size = nInternalCells*5
    start = mpi.exscan(size)
    end = mpi.scan(size)
    size = mpi.bcast(end, root=nProcessors-1)
    return cellStart, cellEnd, start, end, size, mpi

def getInternalFields(case, time, fieldFile):
    import h5py
    time = float(time)
    cellStart, cellEnd, start, end, size, mpi = getParallelInfo()
    fields = []
    with h5py.File(case + getTime(time) + '.hdf5', 'r', driver='mpio', comm=mpi) as phi:
        for name in fieldNames:
            fields.append(phi[name + '/field'][cellStart:cellEnd])
    fields = [x/y for x, y in zip(fields, reference)]
    field = np.hstack(fields).ravel()
    with h5py.File(fieldFile, 'w', driver='mpio', comm=mpi) as handle:
        fieldData = handle.create_dataset('field', shape=(size,), dtype=field.dtype)
        fieldData[start:end] = field
    return

def writeFields(fieldFile, caseDir, ntime):
    import h5py
    ntime = float(ntime)
    cellStart, cellEnd, start, end, size, mpi = getParallelInfo()
    with h5py.File(fieldFile, 'r', driver='mpio', comm=mpi) as handle:
        fields = handle['field'][start:end]
    fields = fields.reshape((fields.shape[0]/5, 5))
    fields = fields[:,[0]], fields[:,1:4], fields[:,[4]]
    fields = [x*y for x, y in zip(fields, reference)]
    timeFile = caseDir + getTime(ntime) + '.hdf5' 
    shutil.copy(case + stime + '.hdf5', timeFile)
    with h5py.File(timeFile, 'r+', driver='mpio', comm=mpi) as phi:
        for index, name in enumerate(fieldNames):
            field = phi[name + '/field']
            field[cellStart:cellEnd] = fields[index]
            phi[name + '/field'][:] = field
    return

def getHostDir(run_id):
    return '{}/temp/{}/'.format(case, run_id)

def spawnJob(exe, args, **kwargs):
    global cobalt
    corner = cobalt.get_corner()
    returncode = subprocess.call(['runjob', '-n', str(nProcessors), 
                       '-p', str(nProcsPerNode),
                       '--block', cobalt.partition,
                       '--corner', corner,
                       '--shape', subBlockShape,
                       '--exp-env', 'PYTHONPATH',
                       '--verbose', 'INFO',
                       ':', exe] + args, **kwargs)
    cobalt.free_corner(corner)
    #returncode = subprocess.call(['mpirun', '-np', str(nProcessors), exe] + args, **kwargs)
    return returncode

def runCase(initFields, parameter, nSteps, run_id, interprocess):
    cobalt.interprocess = interprocess

    # generate case folders
    caseDir = getHostDir(run_id)
    if not os.path.exists(caseDir):
        os.makedirs(caseDir)
    shutil.copy(case + 'mesh.hdf5', caseDir)
    for pkl in glob.glob(case + '*.pkl'):
        shutil.copy(pkl, caseDir)

    
    # write initial field
    if spawnJob(python, [fileName, 'RUN', 'writeFields', initFields, caseDir, str(time)]):
        raise Exception('initial field conversion failed')

    # modify problem file
    shutil.copy(case + problem, caseDir)
    problemFile = caseDir + problem
    with open(problemFile, 'r') as f:
        lines = f.readlines()
    with open(problemFile, 'w') as f:
        for line in lines:
            writeLine = line.replace('NSTEPS', str(nSteps))
            writeLine = writeLine.replace('STARTTIME', str(time))
            writeLine = writeLine.replace('CASEDIR', '\'{}\''.format(caseDir))
            writeLine = writeLine.replace('PARAMETER', str(parameter))
            f.write(writeLine)

    outputFile = caseDir  + 'output.log'
    with open(outputFile, 'w') as f:
        #if spawnJob(python, [problemFile], stdout=f, stderr=f):
        if spawnJob(python, [program, problemFile, '--mira', '-n', '--coresPerNode', str(nProcsPerNode), '--unloadingStages', '4'], stdout=f, stderr=f):
            raise Exception('Execution failed, check error log:', outputFile)

    # read final fields
    times = [float(x[:-5]) for x in os.listdir(caseDir) if isfloat(x[:-5]) and x.endswith('.hdf5')]
    lastTime = sorted(times)[-1]
    finalFields = caseDir + 'output.h5'
    if spawnJob(python, [fileName, 'RUN', 'getInternalFields', caseDir, str(lastTime), finalFields]):
        raise Exception('final field conversion failed')

    # read objective values
    objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')
    print caseDir

    cobalt.interprocess = None
    return finalFields, objectiveSeries[:-1]

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'RUN':
        func = locals()[sys.argv[2]]
        args = sys.argv[3:]
        func(*args)
    else:
        from fds.cobalt import CobaltManager
        cobalt = CobaltManager(subBlockShape, nRuns)
        cobalt.boot_blocks()

        init = getHostDir('init')
        if not os.path.exists(init):
            os.makedirs(init)
        u0 = init + 'init.h5'
        if spawnJob(sys.executable, [fileName, 'RUN', 'getInternalFields', case, str(time), u0]):
            raise Exception('final field conversion failed')

        #runCase(u0, parameter, steps, 'random', None)
        from fds import shadowing
        shadowing(runCase, u0, parameter, dims, segments, steps, 0, simultaneous_runs=nRuns, get_host_dir=getHostDir, spawn_compute_job=spawnJob)

        cobalt.free_blocks()
