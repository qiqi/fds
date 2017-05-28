# this file serves to test if some short commands are legitimate,
# since there is no ipython on hypersonic

from numpy import *
import sys
sys.path.append('/scratch/niangxiu/fds/')
import fds
from matplotlib.pyplot import *

nLE = 33
cp = fds.checkpoint.load_last_checkpoint('charles_3', nLE)
fds.compute.run_compute([cp.v])
# print(dir(cp))
# print(cp.V[1])
print(cp.v.shape)
