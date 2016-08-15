from __future__ import division
import os
import copy
import collections

_temp_state_path = os.path.join(os.path.dirname(os.path.abspath(__file__))
                                'temp_state_dir')

def clear_temp_states():
    shutil.rmtree(_temp_state_path)

def TempPrimalState(name, mpi_run_cmd, mpi_read, mpi_write):
    if not os.path.exists(_temp_state_path):
        os.mkdir(_temp_state_path)
    name = os.path.join(_temp_state_path, name)
    return PrimalState(name, mpi_run_cmd, mpi_read, mpi_write)

class PrimalState:
    def __init__(self, name, mpi_run_cmd, mpi_read, mpi_write):
        self.name = name
        self.mpi_run_cmd = mpi_run_cmd
        self.mpi_read = mpi_read
        self.mpi_write = mpi_write

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, PrimalState) and self.name == other.name

    def __ne__(self, other):
        return not(self == other)

    def __rsub__(self, other):
        assert isinstance(other, PrimalState)
        return TangentState(self, {other: 1})

    def __add__(self, tangent):
        assert isinstance(tangent, TangentState)
        assert tangent.base_state == self and tangent.evaluated
        return tangent.perturbed_state

class TangentState:
    def __init__(self, base_state, perturbations={}):
        self.base_state = base_state
        self.perturbations = collections.defaultdict(float)
        for key, val in perturbations:
            self.perturbations[key] += val

    @property
    def evaluated(self):
        c = self.perturbations.values()
        return len(c) == 0 or (len(c) == 1 and c[0] == 1)

    NOT_EVALUATED = None

    @property
    def perturbed_state(self):
        if self.evaluated and len(self.perturbations):
            return self.perturbations.keys()[0]
        elif self.evaluated:
            return self.base_state
        else:
            return self.NOT_EVALUATED

    def __neg__(self):
        neg_self = TangentState(self.base_state)
        for key, val in self.perturbations:
            neg_self.perturbations[key] -= val
        return neg_self

    def __add__(self, other):
        if isinstance(other, PrimalState):
            return other + self
        assert isinstance(other, TangentState)
        assert other.base_state == self.base_state
        result = TangentState(self.base_state)
        for key, val in self.perturbations:
            result.perturbations[key] += val
        for key, val in other.perturbations:
            result.perturbations[key] += val
        return result

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, c):
        result = TangentState(self.base_state)
        for key, val in self.perturbations:
            result.perturbations[key] += val * c
        return result

    def __rmul__(self, c):
        return self.__mul__(c)

    def __truediv__(self, c):
        return self * (1./c)
