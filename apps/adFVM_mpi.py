import os
import sys
import subprocess
import numpy as np
import shutil
import glob
import h5py

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
fileName = os.path.abspath(__file__)
python = sys.executable

nProcessors = 16
nRuns = 2
parameter = 1.0
dims = 50
segments = 20
steps = 50

time = 1.0
source = '/home/talnikar/adFVM/'
problem = 'periodic_wake.py'
case = source + 'cases/periodic_wake2/'
#problem = 'cylinder.py'
#case = source + 'cases/cylinder/orig/'

def getTime(time):
    stime = str(time)
    if time.is_integer():
        stime = str(int(time))
    return stime
stime = getTime(time)

fieldNames = ['rho', 'rhoU', 'rhoE']
program = source + 'apps/problem.py'

reference = [1., 200., 2e5]

def getParallelInfo():
    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    rank = mpi.rank 
    
    with h5py.File(case + 'mesh.hdf5', 'r') as mesh:
        nCount = mesh['parallel/end'][rank]-mesh['parallel/start'][rank]
        nInternalCells = nCount[4]
        nGhostCells = nCount[2]-nCount[3]
        nCells = nInternalCells + nGhostCells
        start = mpi.exscan(nCells)
        end = mpi.scan(nCells)
        cells = np.arange(start, end)
        start = 0
        for i in range(0, nProcessors):
            n = nInternalCells[i] 
            internalCells.append(np.arange(start, start + n))
            start += n + nGhostCells[i]

    size = len(cells)*5
    start = mpi.exscan(size)
    end = mpi.scan(size)
    size = mpi.bcast(end, root=nProcessors-1)
    return cells, start, end, size

def getInternalFields(case, time, fieldFile):
    time = float(time)
    cells, start, end, size = getParallelInfo()
    fields = []
    with h5py.File(case + getTime(time) + '.hdf5', 'r') as phi:
        for name in fieldNames:
            fields.append(phi[name + '/field'][cells])
    fields = [x/y for x, y in zip(fields, reference)]
    field = np.hstack(fields).ravel()
    with h5py.File(fieldFile, 'w') as handle:
        handle.create_dataset('field', shape=(size,), field.dtype)
        handle[start:end] = field
    return

def writeFields(fieldFile, caseDir, ntime):
    ntime = float(ntime)
    cells, start, end, size = getParallelInfo()
    with h5py.File(fieldFile, 'r') as handle:
        fields = handle['field'][start:end]
    fields = fields.reshape((fields.shape[0]/5, 5))
    fields = fields[:,[0]], fields[:,1:4], fields[:,[4]]
    fields = [x*y for x, y in zip(fields, reference)]
    timeFile = caseDir + getTime(ntime) + '.hdf5' 
    shutil.copy(case + stime + '.hdf5', timeFile)
    with h5py.File(timeFile, 'r+') as phi:
        for index, name in enumerate(fieldNames):
            field = phi[name + '/field']
            field[cells] = fields[index]
            phi[name + '/field'][:] = field
    return

def getHostDir(run_id):
    return '{}/temp/{}/'.format(case, run_id)

def spawnJob(exe, args, **kwargs):
    return call(['mpirun', '-np', str(nProcessors), exe] + args, **kwargs)

def runCase(initFields, parameters, nSteps, run_id):
    # generate case folders
    caseDir = getHostDir(run_id)
    mesh.case = caseDir
    if not os.path.exists(caseDir):
        os.makedirs(caseDir)
    shutil.copy(case + problem, caseDir)
    shutil.copy(case + 'mesh.hdf5', caseDir)
    for pkl in glob.glob(case + '*.pkl'):
        shutil.copy(pkl, caseDir)

    # write initial field
    if spawnJob(python, [fileName, 'RUN', 'writeFields', caseDir, str(time)]):
        raise Exception('initial field conversion failed')
    writeFields(initFields, caseDir, time)

    # modify problem file
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
        if spawnJob(python, [problemFile], stdout=f, stderr=f):
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

    return finalFields, objectiveSeries[:-1]

if __name__ == '__main__':

    if sys.argv[1] == 'RUN':
        func = locals()[sys.argv[2]]
        args = sys.argv[3:]
        func(*args)
    
    u0 = getHostDir('init') + 'init.h5'
    if spawnJob(sys.executable, [fileName, 'RUN', 'getInternalFields', case, str(time), u0]):
        raise Exception('final field conversion failed')

    #runCase(u0, parameters, steps, 'random')
    from fds import shadowing
    shadowing(runCase, u0, parameter, dims, segments, steps, 0, simultaneous_runs=nRuns
            get_host_dir=getCaseDir, spawn_compute_job=spawnJob)
