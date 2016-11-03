import os
import sys
import shutil
import tempfile
import subprocess

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

test_path = os.path.join(my_path, 'test_autonomous_system', 'openfoam4')
foam_path = os.path.join(my_path, '../tools/openfoam4/pisoFoam')
tool_path = os.path.join(my_path, '../tools/openfoam4/scripts')

test_one_step = '''
rm -rf {0}/one_step
mkdir {0}/one_step
cp -r {0}/../../data/pisofoam_restart/0 {0}/one_step
cp -r {0}/../../data/pisofoam_restart/constant {0}/one_step
cp -r {0}/../../data/pisofoam_restart/system {0}/one_step
sed -i -e "s/endTime         1;/endTime         2;/g" {0}/one_step/system/controlDict
decomposePar -case {0}/one_step > {0}/one_step/out0
mpiexec -np 2 {1}/pisoFoam -parallel -case {0}/one_step > {0}/one_step/out
reconstructPar -case {0}/one_step > {0}/one_step/out1
cp {0}/one_step/2/*.gz {0}/one_step
gunzip {0}/one_step/*.gz
'''.strip().format(test_path, foam_path)

test_two_steps = '''
rm -rf {0}/two_steps
mkdir {0}/two_steps
mkdir {0}/two_steps/step_1
cp -r {0}/../../data/pisofoam_restart/0 two_steps/step_1
mkdir {0}/two_steps/step_2
gzip {0}/two_steps/step_1/0/*
cp -r {0}/../../data/pisofoam_restart/constant {0}/two_steps/step_1
cp -r {0}/../../data/pisofoam_restart/system {0}/two_steps/step_1
decomposePar -case {0}/two_steps/step_1 > {0}/two_steps/step_1/out0
mpiexec -np 2 {1}/pisoFoam -parallel -case {0}/two_steps/step_1 > {0}/two_steps/step_1/out
mpiexec -np 2 python {2}/foam_to_h5.py {0}/two_steps/step_1 1 {0}/two_steps/1.hdf5
mpiexec -np 2 python {2}/h5_to_foam.py two_steps/step_1 {0}/two_steps/1.hdf5 {0}/two_steps/step_2 0
mpiexec -np 2 {1}/pisoFoam -parallel -case {0}/two_steps/step_2 > {0}/two_steps/step_2/out
reconstructPar -case {0}/two_steps/step_2 > {0}/two_steps/step_2/out1
cp {0}/two_steps/step_2/1/*.gz {0}/two_steps/step_2
gunzip {0}/two_steps/step_2/*.gz
'''.strip().format(test_path, foam_path, tool_path)

compare = '''
diff one_step/U two_steps/
diff one_step/nut two_steps/
diff one_step/nuTilda two_steps/
diff one_step/nuTilda_0 two_steps/
diff one_step/p two_steps/
diff one_step/phi two_steps/
diff one_step/phi_0 two_steps/
diff one_step/U_0 two_steps/
'''.strip()

#if __name__ == '__main__':
def test_pisoform4_io():
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)
    subprocess.check_call(' && '.join(test_one_step.splitlines()),
                          shell=True, cwd=test_path)
    subprocess.check_call(' && '.join(test_two_steps.splitlines()),
                          shell=True, cwd=test_path)
    for comp in compare.splitlines():
        p = subprocess.Popen(comp, shell=True, cwd=test_path,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        assert len(out.splitlines()) <= 5
