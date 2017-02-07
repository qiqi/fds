#!/usr/bin/python2 -u
import os
import subprocess
import numpy as np
import shutil
import glob
import h5py
import sys
import cPickle as pickle
sys.setrecursionlimit(10000)

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

nProcessors = 16
parameter = 1.0
nRuns = 4
dims = 120
segments = 20
steps = 5000
adj_write_interval = 5000
#nRuns = 1
#dims = 2
#segments = 5
#steps = 2
#adj_write_interval = 2

time = 1.0
source = '/master/home/talnikar/adFVM/'
problem = 'periodic_wake.py'
case = '/scratch/talnikar/periodic_wake_adj/'
#case = '/scratch/talnikar/periodic_wake_test/'

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

fieldNames = ['rhoa', 'rhoUa', 'rhoEa']
program = source + 'apps/adjoint.py'

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
    with h5py.File(timeFile, 'r+') as phi:
        for index, name in enumerate(fieldNames):
            field = phi[name + '/field'][:]
            field[internalCells] = fields[index]
            phi[name + '/field'][:] = field

simTimes = []
for hdf in glob.glob(case + '*.hdf5'):
    if 'mesh' not in hdf:
	simTimes.append(os.path.basename(hdf))
simTimes.sort(key=lambda x: float(x[:-5]))

def spawnJob(exe, args, **kwargs):
    from fds.slurm import grab_from_SLURM_NODELIST
    interprocess = kwargs['interprocess']
    del kwargs['interprocess']
    #nodes = grab_from_SLURM_NODELIST(1, interprocess)
    #print('spawnJob', nodes, exe, args)
    #returncode = subprocess.call(['mpirun', '--host', ','.join(nodes.grabbed_nodes)
    #                   , exe] + args, **kwargs)
    #nodes.release()
    returncode = subprocess.call(['mpirun', '-np', str(nProcessors), exe] + args, **kwargs)
    return returncode

def runCase(initFields, nSteps, segment, run_id, interprocess):

    # generate case folders
    caseDir = '{}/temp/{}/'.format(case, run_id)
    mesh.case = caseDir
    if not os.path.exists(caseDir):
        os.makedirs(caseDir)
    shutil.copy(case + problem, caseDir)
    shutil.copy(case + 'mesh.hdf5', caseDir)
    jump = nSteps/adj_write_interval
    start = len(simTimes) - 2 - jump*segment
    times = simTimes[start:start + jump + 1]
    for stime in times:
	shutil.copy(case + stime, caseDir)

    for pkl in glob.glob(case + '*.pkl'):
        shutil.copy(pkl, caseDir)

    time_data = np.loadtxt(case + '{}.{}.txt'.format(segments*nSteps, adj_write_interval))
    time_data = time_data[start*nSteps: (start + 1)*nSteps]
    np.savetxt(caseDir + '{}.{}.txt'.format(nSteps, adj_write_interval), time_data)

    # write initial field
    ntime = float(times[-1][:-5])
    writeFields(initFields, caseDir, ntime)

    # modify problem file
    problemFile = caseDir + problem
    with open(problemFile, 'r') as f:
        lines = f.readlines()
    with open(problemFile, 'w') as f:
        for line in lines:
            writeLine = line.replace('NSTEPS', str(nSteps))
            writeLine = writeLine.replace('STARTTIME', times[0][:-5])
            writeLine = writeLine.replace('CASEDIR', '\'{}\''.format(caseDir))
            f.write(writeLine)

    outputFile = caseDir  + 'output.log'
    with open(outputFile, 'w') as f:
        if spawnJob(sys.executable, [program, problemFile], stdout=f, stderr=f, interprocess=interprocess):
     	    raise Exception('Execution failed, check error log:', outputFile)

    # read final fields
    lastTime = float(times[0][:-5])
    finalFields = getInternalFields(caseDir, lastTime)
    # read objective values
    print caseDir

    return finalFields

from multiprocessing import Manager, Pool
manager = Manager()
interprocess = [manager.Lock(), manager.dict()]

if __name__ == '__main__':
    u0 = getInternalFields(case, time)
    #runCase(u0, parameters, steps, 'random')
    V = np.random.rand(u0.shape[0], dims)
    Rs = []
    for i in range(0, segments):
        Vn = []
        res = []
        print i
	threads = Pool(nRuns)
        for j in range(0, dims):
            run_id = 'segment{}_perturb{}'.format(i,j)
	    v0 = V[:,j]
            res.append(threads.apply_async(runCase, (v0, steps, i, run_id, interprocess)))
        for j in range(0, dims):
            Vn.append(res[j].get())
        V = np.array(Vn).T
        Q, R = np.linalg.qr(V)
        V = Q[:]
	Rs.append(R)
	threads.close()

        with open(case + '/checkpoint/m{}_{}'.format(dims, i+1), 'w') as f:
	    pickle.dump([V, Rs], f)
	
