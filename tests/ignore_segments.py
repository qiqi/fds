from numpy import *

data = load('lss.npz')
R = data['R']
i = arange(R.shape[1])
print('Lyapunov exponents:')
for i, lyapunov in enumerate(log(abs(R[5:,i,i])).mean(0)):
    print(i, lyapunov)
