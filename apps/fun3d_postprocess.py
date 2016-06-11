from __future__ import print_function

import os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(my_path)
from fun3d import *

checkpoint = most_recent_checkpoint(M_MODES)


data = load('lss.npz')
R = data['R']
i = arange(R.shape[1])
print('Lyapunov exponents:')
for i, lyapunov in enumerate(log(abs(R[:,i,i])).mean(0)):
    print(i, lyapunov)
