import copy
import string

import numpy as np

from ..symbolic_value import _is_like_sa_value, symbolic_array_value

# ============================================================================ #
#                                   Op class                                   #
# ============================================================================ #

class OpBase(object):
    '''
    Perform operations between values, and remember whether the operation
    accesses grid neighbors

    Op(operation, inputs)
        operation: a function that takes a list of inputs as arguments
        inputs: a list of symbolic array
    '''
    def __init__(self, py_operation, inputs, access_neighbor=False,
                 shape_keeper=np.ones, shapes=None, name=None):
        self.py_operation = py_operation
        self.inputs = []
        for inp in inputs:
            if _is_like_sa_value(inp):
                self.inputs.append(inp)
            else:
                try:
                    inp = np.array(inp, np.float64)
                except (IndexError, TypeError):
                    pass
                self.inputs.append(inp)
        self.name = name

        def produce_shape_keepers(a):
            if _is_like_sa_value(a):
                return shape_keeper(a.shape)
            else:
                return np.array(a, np.float64)
        #print name, produce_shape_keepers(self.inputs[0])
        shape_keeper_inputs = tuple(map(produce_shape_keepers, self.inputs))
        if shapes is None:
            shape_outputs = py_operation(*shape_keeper_inputs)
            if isinstance(shape_outputs, tuple):
                shapes = tuple([output.shape for output in shape_outputs])
            else:
                shapes = (shape_outputs.shape,)
        print name, shape_keeper_inputs, shapes

        self.access_neighbor = access_neighbor
        self.outputs = tuple([symbolic_array_value(shape, self) for shape in shapes])
        if len(shapes) == 1:
            self.output = self.outputs[0]

    def perform(self, input_objects):
        assert len(input_objects) == len(self.inputs)
        return self.py_operation(*input_objects)

    def __repr__(self):
        return 'Operator {0}'.format(self.name)
