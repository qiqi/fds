import os
import sys
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

from fds import timedilation

def factorial(k):
    return 1.0 if k <= 1 else k * factorial(k-1)

def test_monomial():
    for k in range(2, 10):
        x = arange(k+1) ** k / factorial(k)
        assert abs(timedilation.compute_dxdt(x)) < 1E-12

#def test_sine():
# if __name__ == '__main__':
#     dydx = []
#     for k in range(2, 10):
#         dydx.append([])
#         for dx in [0.1, 0.01, 0.001, 0.0001]:
#             y = sin(arange(k + 1) * dx)
#             dydx[-1].append(timedilation.compute_dxdt(y) / dx)
#     error = array(dydx) - 1
