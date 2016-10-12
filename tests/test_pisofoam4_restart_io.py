import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

test_one_step = '''
rm -rf one_step
mkdir one_step
cp -r data/0 one_step
cp -r data/constant one_step
cp -r data/system one_step
cd one_step
sed -i -e "s/endTime         1;/endTime         2;/g" system/controlDict
decomposePar > out0
mpiexec -np 4 ../../pisoFoam/pisoFoam -parallel > out
reconstructPar
cp 2/*.gz .
gunzip *.gz
cd ..
'''

test_two_steps = '''
rm -rf two_steps
mkdir two_steps
mkdir two_steps/step_1
mkdir two_steps/step_2
cp -r data/0 two_steps/step_1
gzip two_steps/step_1/0/*
cp -r data/constant two_steps/step_1
cp -r data/system two_steps/step_1
cd two_steps/step_1
decomposePar > out0
mpiexec -np 4 ../../../pisoFoam/pisoFoam -parallel > out
cd ../..
mpiexec -np 4 python foam_to_h5.py two_steps/step_1 1 two_steps/1.hdf5
mpiexec -np 4 python h5_to_foam.py two_steps/step_1 two_steps/1.hdf5 two_steps/step_2 0
cd two_steps/step_2
mpiexec -np 4 ../../../pisoFoam/pisoFoam -parallel > out
reconstructPar > out1
cp 1/*.gz ..
gunzip ../*.gz
cd ../..
'''

compare = '''
diff one_step/U two_steps/
diff one_step/nut two_steps/
diff one_step/nuTilda two_steps/
diff one_step/nuTilda_0 two_steps/
diff one_step/p two_steps/
diff one_step/phi two_steps/
diff one_step/phi_0 two_steps/
diff one_step/U_0 two_steps/
'''

#if __name__ == '__main__':
def test_pisoform4_io():
    pass
