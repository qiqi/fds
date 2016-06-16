import os
import sys
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import timedilation

def factorial(k):
    return 1 if k <= 1 else k * factorial(k-1)

def test_monomial():
    for k in range(2, 10):
        x = arange(k+1) ** k / factorial(k)
        timedilation.set_order_of_accuracy(k)
        assert abs(timedilation.compute_dxdt(x)) < 1E-12
