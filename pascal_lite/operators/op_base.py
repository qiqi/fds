import copy
import string

import numpy as np

from ..symbolic_value import _is_like_sa_value, symbolic_array_value

# ============================================================================ #
#                                   Op class                                   #
# ============================================================================ #

class OpBase(object):
    '''
    Perform operations between values
    Op(operation, inputs)
        operation: a function that takes a list of inputs as arguments
        inputs: a list of symbolic array
    '''
    def __init__(self, py_operation, inputs, shape_keeper=np.ones,
                 output_shapes=None, are_outputs_distributed=None, name=None):
        self.py_operation = py_operation
        self.inputs = []
        for inp in inputs:
            if _is_like_sa_value(inp):
                self.inputs.append(inp)
                if are_outputs_distributed is None:
                    are_outputs_distributed = inp.is_distributed
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

        shape_keeper_inputs = tuple(map(produce_shape_keepers, self.inputs))
        if output_shapes is None:
            dummy_outputs = py_operation(*shape_keeper_inputs)
            if isinstance(dummy_outputs, tuple):
                output_shapes = tuple([out.shape for out in dummy_outputs])
            else:
                output_shapes = (dummy_outputs.shape,)

        if are_outputs_distributed is True or are_outputs_distributed is False:
            are_outputs_distributed = [are_outputs_distributed
                                       for s in output_shapes]
        self.outputs = tuple([
            symbolic_array_value(shape, self, is_distributed=is_dist)
            for shape, is_dist in zip(output_shapes, are_outputs_distributed)
        ])
        if len(output_shapes) == 1:
            self.output = self.outputs[0]  # alias

    def perform(self, input_objects):
        assert len(input_objects) == len(self.inputs)
        output = self.py_operation(*input_objects)
        return output if isinstance(output, tuple) else (output,)

    def __repr__(self):
        return 'Operator {0}'.format(self.name)
