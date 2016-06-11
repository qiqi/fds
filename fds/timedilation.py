from numpy import *

class TimeDilation:
    def __init__(self, run, u0, parameter, run_id, interprocess):
        dof = u0.size
        u0p, _ = run(u0, parameter, 1, run_id, interprocess)
        self.dxdt = (u0p - u0)   # time step size turns out cancelling out
        self.dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)

    def contribution(self, v):
        return dot(self.dxdt, v) / (self.dxdt**2).sum()

    def project(self, v):
        dv = outer(self.dxdt_normalized, dot(self.dxdt_normalized, v))
        return v - dv.reshape(v.shape)
