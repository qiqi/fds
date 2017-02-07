#!/usr/bin/python2 -u
import os
import subprocess
import numpy as np
import shutil
import glob
import h5py
import sys
sys.setrecursionlimit(10000)

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

nProcessors = 1
nRuns = 64
parameter = 1.0
dims = 120
segments = 20
steps = 5000

time = 1.0
source = '/master/home/talnikar/adFVM/'
problem = 'periodic_wake.py'
case = source + 'cases/periodic_wake/'
#problem = 'cylinder.py'
#case = source + 'cases/cylinder/orig/'

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

fieldNames = ['rho', 'rhoU', 'rhoE']
program = source + 'apps/problem.py'

reference = [1., 200., 2e5]
def getInternalFields(case, time):
    fields = []
    with h5py.File(case + getTime(time) + '.hdf5', 'r') as phi:
        for name in fieldNames:
            fields.append(phi[name + '/field'][:][internalCells])
    fields = [x/y for x, y in zip(fields, reference)]
    return np.hstack(fields).ravel()

def writeFields(fields, caseDir, ntime):
    fields = fields.reshape((fields.shape[0]/5, 5))
    fields = fields[:,[0]], fields[:,1:4], fields[:,[4]]
    fields = [x*y for x, y in zip(fields, reference)]
    timeFile = caseDir + getTime(ntime) + '.hdf5' 
    shutil.copy(case + stime + '.hdf5', timeFile)
    with h5py.File(timeFile, 'r+') as phi:
        for index, name in enumerate(fieldNames):
            field = phi[name + '/field'][:]
            field[internalCells] = fields[index]
            phi[name + '/field'][:] = field

def runCase(initFields, parameters, nSteps, run_id):

    # generate case folders
    caseDir = '{}/temp/{}/'.format(case, run_id)
    mesh.case = caseDir
    if not os.path.exists(caseDir):
        os.makedirs(caseDir)
    shutil.copy(case + problem, caseDir)
    shutil.copy(case + 'mesh.hdf5', caseDir)
    for pkl in glob.glob(case + '*.pkl'):
        shutil.copy(pkl, caseDir)

    # write initial field
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
    for rep in range(0, 5):
        try:
            with open(outputFile, 'w') as f:
                returncode = subprocess.call(['srun', '--exclusive', '-n', str(nProcessors),
                                  '-N', '1', '--resv-ports',
                                  program, problemFile, '--voyager'],
                                  stdout=f, stderr=f)
            if returncode:
                raise Exception('Execution failed, check error log:', outputFile)
            objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')
            break 
        except Exception as e:
            print caseDir, 'rep', rep, str(e)
            import time as timer
            timer.sleep(2)

    # read final fields
    times = [float(x[:-5]) for x in os.listdir(caseDir) if isfloat(x[:-5]) and x.endswith('.hdf5')]
    lastTime = sorted(times)[-1]
    finalFields = getInternalFields(caseDir, lastTime)
    # read objective values
    print caseDir

    return finalFields, objectiveSeries

if __name__ == '__main__':
    u0 = getInternalFields(case, time)
    #runCase(u0, parameters, steps, 'random')
    from fds import shadowing, continue_shadowing
    from fds.checkpoint import *
    #checkpoint = load_last_checkpoint(case + '/checkpoint', dims)
    #continue_shadowing(runCase, parameter, checkpoint, segments, steps, simultaneous_runs=nRuns,
#        checkpoint_path=case + '/checkpoint')
    shadowing(runCase, u0, parameter, dims, segments, steps, 0, simultaneous_runs=nRuns, checkpoint_path=case + '/checkpoint')
