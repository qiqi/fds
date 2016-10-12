import os
import sys
import shutil
import tempfile
import subprocess

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

test_path = os.path.join(my_path, 'test_autonomous_system', 'openfoam4')

test_one_step = '''
rm -rf one_step
mkdir one_step
cp -r ../../data/pisofoam_restart/0 one_step
cp -r ../../data/pisofoam_restart/constant one_step
cp -r ../../data/pisofoam_restart/system one_step
cd one_step
sed -i -e "s/endTime         1;/endTime         2;/g" system/controlDict
decomposePar > out0
mpiexec -np 2 ../../../../tools/openfoam4/pisoFoam/pisoFoam -parallel > out
reconstructPar > out1
cp 2/*.gz .
gunzip *.gz
cd ..
'''.strip()

test_two_steps = '''
rm -rf two_steps
mkdir two_steps
mkdir two_steps/step_1
cp -r ../../data/pisofoam_restart/0 two_steps/step_1
mkdir two_steps/step_2
gzip two_steps/step_1/0/*
cp -r ../../data/pisofoam_restart/constant two_steps/step_1
cp -r ../../data/pisofoam_restart/system two_steps/step_1
cd two_steps/step_1
decomposePar > out0
mpiexec -np 2 ../../../../../tools/openfoam4/pisoFoam/pisoFoam -parallel > out
cd ../..
mpiexec -np 2 python ../../../tools/openfoam4/scripts/foam_to_h5.py two_steps/step_1 1 two_steps/1.hdf5
mpiexec -np 2 python ../../../tools/openfoam4/scripts/h5_to_foam.py two_steps/step_1 two_steps/1.hdf5 two_steps/step_2 0
cd two_steps/step_2
mpiexec -np 2 ../../../../../tools/openfoam4/pisoFoam/pisoFoam -parallel > out
reconstructPar > out1
cp 1/*.gz ..
gunzip ../*.gz
cd ../..
'''.strip()

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
