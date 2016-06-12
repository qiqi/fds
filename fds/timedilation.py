import sys
from numpy import *
from multiprocessing import Pool

def compute_dxdt(u):
    '''
    time step size turns out cancelling out
    '''
    assert len(u) == 3 # second order accuracy for now
    dxdt_1st_order = u[1] - u[0]
    dxdt_2nd_order = (-u[2] + 4 * u[1] - 3 * u[0]) / 2
    difference = linalg.norm(ravel(dxdt_2nd_order - dxdt_1st_order))
    relative_difference = difference / linalg.norm(ravel(dxdt_2nd_order))
    if relative_difference > 0.01:
        sys.stderr.write('Warning: dxdt in time dilation inaccurate. ')
        sys.stderr.write('Relative error = {0}\n'.format(relative_difference))
    return dxdt_2nd_order

class TimeDilation:
    order_of_accuracy = 2

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
