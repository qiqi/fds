import os
import subprocess
import numpy as np

from pyRCF import RCF
from field import IOField
import h5py

case = 'cases/cylinder/'
program = '/home/talnikar/adFVM/problem.py'
problem = 'cylinder.py'
nProcessors = 2
time = 0.0
solver = RCF(case)
mesh = solver.mesh

def runCase(initFields, parameters, nSteps, runID, _):

    # generate case folders
    case = case + '/'
    caseDir = '{}{}/'.format(case, runID)
    os.mkdir(caseDir)
    shutil.copyFile(case + problem, caseDir)
    shutil.copyTree(case + 'mesh.hdf5', caseDir)
    shutil.copyTree(case + '0.hdf5', caseDir)
    problem = caseDir + problem

    # write initial field
    #initFields = distributeData(initGlobalFields)
    fields = solver.unstackFields(initFields, IOField)
    fields = solver.primitive(*fields)
    mesh.case = case
    IOField.openHandle(time)
    for phi in fields:
        phiO = IOField.read(phi.name)
        phiO.field[:mesh.nInternalCells] = phi.field
    IOField.closeHandle()
    mesh.case = caseDir
    IOField.openHandle(time)
    for phi in fields:
        phi.write()
    IOField.closeHandle()

    # modify problem.py
    with open(problem, 'r') as f:
        lines = f.readlines()
    with open(problem, w) as f:
        for line in lines:
            writeLine = line.replace('NSTEPS', str(nSteps))
            writeLine = line.replace('PARAMETERS', parameters)
            f.write(writeLine)

    subprocess.Popen(['mpirun', '-np', str(nProcessors),
                      program, problem])

    # read final fields
    fields = solver.initFields(lastTime)
    fields = solver.stackFields(fields, np)
    finalFields = fields[:mesh.nInternalCells]

    # read objective values
    objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')

    return finalFields, objectiveSeries

from fds import shadowing
