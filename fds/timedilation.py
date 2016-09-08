from __future__ import division
import sys
import numpy as np
import pascal_lite as pascal
from multiprocessing import Pool

def set_order_of_accuracy(order_of_accuracy):
    if order_of_accuracy < 2:
        order_of_accuracy = 2
        sys.stderr.write('Order of accuracy too low, setting to 2 instead\n')
    TimeDilation.order_of_accuracy = order_of_accuracy

def compute_dxdt_of_order(u, order):
    assert order >= 1
    A = np.array([np.arange(order + 1) ** i for i in range(order + 1)])
    b = np.zeros(order + 1)
    b[1] = 1
    c = np.linalg.solve(A, b)
    return sum([c[i]*u[i] for i in range(0, order+1)])
    #return pascal.dot(c, u[:order+1])

def compute_dxdt(u):
    '''
    time step size turns out cancelling out
    '''
    dxdt_higher_order = compute_dxdt_of_order(u, len(u) - 1)
    dxdt_lower_order = compute_dxdt_of_order(u, len(u) - 2)
    ravel = lambda x: x
    difference = pascal.norm(ravel(dxdt_higher_order - dxdt_lower_order))
    #relative_difference = difference / pascal.norm(ravel(dxdt_higher_order))
    #if relative_difference > 0.01:
    #    sys.stderr.write('Warning: dxdt in time dilation inaccurate. ')
    #    sys.stderr.write('Relative error = {0}\n'.format(relative_difference))
    return dxdt_higher_order

class TimeDilationBase:
    def contribution(self, v):
        if self.dxdt is None:
            if len(v.shape) == 0:
                return pascal.dot(v, v*0)
            else:
                return pascal.dot(v, v[0]*0)
        else:
            return pascal.dot(v, self.dxdt) / pascal.dot(self.dxdt, self.dxdt)

    def project(self, v):
        if self.dxdt_normalized is None:
            return v
        else:
            dv = pascal.outer(pascal.dot(v, self.dxdt_normalized), self.dxdt_normalized)
            return v - dv.reshape(v.shape)

class TimeDilationExact(TimeDilationBase):
    def __init__(self, run_ddt, u0, parameter):
        if run_ddt is not 0:
            self.dxdt = run_ddt(u0, parameter)
            self.dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)
        else:
            self.dxdt = None
            self.dxdt_normalized = None

class TimeDilation(TimeDilationBase):
    order_of_accuracy = 3

    def __init__(self, run, u0, parameter, run_id,
                 simultaneous_runs, interprocess):
        threads = Pool(simultaneous_runs)
        res = []
        for steps in range(1, self.order_of_accuracy + 1):
            run_id_steps = run_id + '_{0}steps'.format(steps)
            res.append(threads.apply_async(
                run, (u0.field, parameter, steps, run_id_steps, interprocess)))
        
        u = [res_i.get()[0] for res_i in res]
        u = [u0] + [pascal.symbolic_array(field=ui) for ui in u]

        threads.close()
        threads.join()
        self.dxdt = compute_dxdt(u)
        self.dxdt_normalized = self.dxdt / pascal.norm(self.dxdt)
