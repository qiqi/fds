import os
import sys
import subprocess
import numpy as np
import shutil
import glob
import h5py

def isfloat(s):
    try:
        f = float(s)
        return True
    except ValueError:
        return False

source = '/home/talnikar/adFVM/'
problem = 'periodic_wake.py'
nProcessors = 4
nRuns = 2
time = 2.0

fieldNames = ['rho', 'rhoU', 'rhoE']
case = source + 'cases/periodic_wake/'
program = source + 'apps/problem.py'

def getTime(time):
    stime = str(time)
    if time.is_integer():
        stime = str(int(time))
    return stime
stime = getTime(time)

internalCells = []
with h5py.File(case + 'mesh.hdf5', 'r') as mesh:
    nCount = mesh['parallel/end'][:]-mesh['parallel/start'][:]
    nInternalCells = nCount[:,4]
    nGhostCells = nCount[:,2]-nCount[:,3]
    start = 0
    for i in range(0, nProcessors):
        n = nInternalCells[i] 
        internalCells.append(np.arange(start, start + n))
        start += n + nGhostCells[i]
internalCells = np.concatenate(internalCells)

def getInternalFields(case, time):
    fields = []
    with h5py.File(case + getTime(time) + '.hdf5', 'r') as phi:
        for name in fieldNames:
            fields.append(phi[name + '/field'][:][internalCells])
    return np.hstack(fields).ravel()

def runCase(initFields, parameters, nSteps, run_id):

    # generate case folders
    caseDir = '{}/temp/{}/'.format(case, run_id)
    mesh.case = caseDir
    if not os.path.exists(caseDir):
        os.makedirs(caseDir)
    shutil.copy(case + problem, caseDir)
    shutil.copy(case + 'mesh.hdf5', caseDir)
    shutil.copy(case + stime + '.hdf5', caseDir)
    for pkl in glob.glob(case + '*.pkl'):
        shutil.copy(pkl, caseDir)

    # write initial field
    initFields = initFields.reshape((initFields.shape[0]/5, 5))
    fields = initFields[:,[0]], initFields[:,1:4], initFields[:,[4]]
    with h5py.File(caseDir + stime + '.hdf5', 'r+') as phi:
        for index, name in enumerate(fieldNames):
            field = phi[name + '/field'][:]
            field[internalCells] = fields[index]
            phi[name + '/field'][:] = field

    # modify problem file
    problemFile = caseDir + problem
    with open(problemFile, 'r') as f:
        lines = f.readlines()
    with open(problemFile, 'w') as f:
        for line in lines:
            writeLine = line.replace('NSTEPS', str(nSteps))
            writeLine = writeLine.replace('STARTTIME', str(time))
            writeLine = writeLine.replace('CASEDIR', '\'{}\''.format(caseDir))
            writeLine = writeLine.replace('PARAMETERS', str(parameters))
            f.write(writeLine)

    outputFile = caseDir  + 'output.log'
    with open(outputFile, 'w') as f:
        returncode = subprocess.call(['mpirun', '-np', str(nProcessors),
                          program, problemFile, 'orig'],
                          stdout=f, stderr=f)
    if returncode:
        raise Exception('Execution failed, check error log:', outputFile)

    # read final fields
    times = [float(x[:-5]) for x in os.listdir(caseDir) if isfloat(x[:-5]) and x.endswith('.hdf5')]
    lastTime = sorted(times)[-1]
    finalFields = getInternalFields(caseDir, lastTime)
    # read objective values
    objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')
    print caseDir

    return finalFields, objectiveSeries[:-1]

from fds import shadowing

u0 = getInternalFields(case, time)
parameters = 100.0
dims = 50
segments = 20
steps = 50
#runCase(u0, parameters, steps, 'random')
shadowing(runCase, u0, parameters, dims, segments, steps, 0, simultaneous_runs=nRuns)
