import os
import sys
import subprocess
import numpy as np
import shutil
import h5py

from adFVM import config
config.hdf5 = True
from adFVM.field import IOField
try:
    from pyRCF import RCF
except:
    print('add adFVM.git/apps to PYTHONPATH')

prefix = '/home/talnikar/adFVM/'
case = prefix + 'cases/cylinder/'
program = prefix + 'problem.py'
problem = 'cylinder.py'
nProcessors = 1
time = 0.0
solver = RCF(case)
mesh = solver.mesh
meshO = mesh.origMesh

def getInternalFields(time):
    fields = solver.initFields(time)
    fields = solver.stackFields(fields, np)
    return fields[:meshO.nInternalCells]

def runCase(initFields, parameters, nSteps, run_id):

    # generate case folders
    caseDir = '{}/temp/{}/'.format(case, run_id)
    mesh.case = caseDir
    if not os.path.exists(caseDir):
        os.mkdir(caseDir)
    problemFile = caseDir + problem
    shutil.copyfile(case + problem, problemFile)
    shutil.copyfile(case + 'mesh.hdf5', caseDir + 'mesh.hdf5')
    shutil.copyfile(case + '0.hdf5', caseDir + '0.hdf5')
    outputFile = caseDir  + 'output.log'

    # write initial field
    #initFields = distributeData(initGlobalFields)
    #sys.stdout = os.devnull
    fields = solver.unstackFields(initFields, IOField)
    fields = solver.primitive(*fields)
    IOField.openHandle(time)
    for phi in fields:
        phiO = IOField.read(phi.name)
        phiO.partialComplete()
        phiO.field[:meshO.nInternalCells] = phi.field
        phiO.write()
    IOField.closeHandle()
    #sys.stdout = sys.__stdout__

    # modify problem.py
    with open(problemFile, 'r') as f:
        lines = f.readlines()
    with open(problemFile, 'w') as f:
        for line in lines:
            writeLine = line.replace('NSTEPS', str(nSteps))
            writeLine = writeLine.replace('STARTTIME', str(time))
            writeLine = writeLine.replace('CASEDIR', '\'{}\''.format(caseDir))
            #writeLine = line.replace('PARAMETERS', parameters)
            f.write(writeLine)

    with open(outputFile, 'w') as f:
        returncode = subprocess.call(['mpirun', '-np', str(nProcessors),
                          program, problemFile, 'orig'],
                          stdout=f, stderr=f)
    if returncode:
        raise Exception('Execution failed, check error log:', outputFile)

    # read final fields
    lastTime = mesh.getTimes()[-1]
    finalFields = getInternalFields(lastTime)
    # read objective values
    objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')

    return finalFields, objectiveSeries

from fds import shadowing

u0 = getInternalFields(time)
parameters = 1.0
dims = 2
segments = 10
steps = 10
#runCase(u0, parameters, steps, 'random')
shadowing(runCase, u0, parameters, dims, segments, steps, 0)
