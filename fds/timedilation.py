import sys
from numpy import *
from multiprocessing import Pool

def set_order_of_accuracy(order_of_accuracy):
    if order_of_accuracy < 2:
        order_of_accuracy = 2
        sys.stderr.write('Order of accuracy too low, setting to 2 instead\n')
    TimeDilation.order_of_accuracy = order_of_accuracy

def compute_dxdt_of_order(u, order):
    assert order >= 1
    A = array([arange(order + 1) ** i for i in range(order + 1)])
    b = zeros(order + 1)
    b[1] = 1
    c = linalg.solve(A, b)
    return dot(c, u[:order+1])

def compute_dxdt(u):
    '''
    time step size turns out cancelling out
    '''
    dxdt_higher_order = compute_dxdt_of_order(u, len(u) - 1)
    dxdt_lower_order = compute_dxdt_of_order(u, len(u) - 2)
    difference = linalg.norm(ravel(dxdt_higher_order - dxdt_lower_order))
    relative_difference = difference / linalg.norm(ravel(dxdt_higher_order))
    if relative_difference > 0.01:
        sys.stderr.write('Warning: dxdt in time dilation inaccurate. ')
        sys.stderr.write('Relative error = {0}\n'.format(relative_difference))
    return dxdt_higher_order

class TimeDilation:
    order_of_accuracy = 3

    def __init__(self, run, u0, parameter, run_id,
                 simultaneous_runs, interprocess):
        threads = Pool(simultaneous_runs)
        res = []
        for steps in range(1, self.order_of_accuracy + 1):
            run_id_steps = run_id + '_{0}steps'.format(steps)
            res.append(threads.apply_async(
                run, (u0, parameter, steps, run_id_steps, interprocess)))
        u = array([u0] + [res_i.get()[0] for res_i in res])
        threads.close()
        threads.join()
        self.dxdt = compute_dxdt(u)
        self.dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)

    def contribution(self, v):
        return dot(self.dxdt, v) / (self.dxdt**2).sum()

    def project(self, v):
        dv = outer(self.dxdt_normalized, dot(self.dxdt_normalized, v))
        return v - dv.reshape(v.shape)
