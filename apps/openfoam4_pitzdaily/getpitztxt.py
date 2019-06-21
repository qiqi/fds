# this file loads the checkpoints in fds
# computes djds v.s. # segments and export to pitz.txt

from numpy import *
import sys
sys.path.append('/home/ubuntu/data/git/fds/')
import fds

cp = fds.checkpoint.load_last_checkpoint('pitzdaily1.1', 16)
c = zeros([200,4])
for i in range(200):
    c[i] = fds.lss_gradient(cp,[0,i+1])
savetxt('pitz11.txt',c)
