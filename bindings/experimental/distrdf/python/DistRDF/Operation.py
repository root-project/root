## @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from __future__ import print_function
from enum import Enum


class Operation(object):
    """
    A Generic representation of an operation. The operation could be a
    transformation or an action.

    Attributes:
        Types: A class member that is an :obj:`Enum` of the types of
            operations supported. This can be either :obj:`ACTION`,
            :obj:`TRANSFORMATION` or :obj:`INSTANT_ACTION`.

        name (str): Name of the current operation.

        args (list): Variable length argument list for the current operation.

        kwargs (dict): Arbitrary keyword arguments for the current operation.

        op_type: The type or category of the current operation
            (:obj:`ACTION`,  :obj:`TRANSFORMATION` or :obj:`INSTANT_ACTION`).

    For the list of operations that your current
    backend supports, try :

    Example::

        import DistRDF
        DistRDF.use(...) # Choose a backend

        print(DistRDF.current_backend.supported_operations)
    """

    Types = Enum("Types", "ACTION TRANSFORMATION INSTANT_ACTION")

    def __init__(self, name, *args, **kwargs):
        """
        Creates a new :obj:`Operation` for the given name
        and arguments.

        Args:
            name (str): Name of the current operation.

        args (list): Variable length argument list for the current
            operation.

        kwargs (dict): Keyword arguments for the current operation.
        """
        self.name = name
        self.args = list(args)
        self.kwargs = kwargs
        self.op_type = self._classify_operation(name)

    def _classify_operation(self, name):
        # Classifies the given operation as action or
        # transformation and returns the type.

        ops = Operation.Types

        operations_dict = {
            'Define': ops.TRANSFORMATION,
            'Filter': ops.TRANSFORMATION,
            'Range': ops.TRANSFORMATION,
            'Aggregate': ops.ACTION,
            'Histo1D': ops.ACTION,
            'Histo2D': ops.ACTION,
            'Histo3D': ops.ACTION,
            'Profile1D': ops.ACTION,
            'Profile2D': ops.ACTION,
            'Profile3D': ops.ACTION,
            'Count': ops.ACTION,
            'Min': ops.ACTION,
            'Max': ops.ACTION,
            'Mean': ops.ACTION,
            'Sum': ops.ACTION,
            'Fill': ops.ACTION,
            'Reduce': ops.ACTION,
            'Report': ops.ACTION,
            'Take': ops.ACTION,
            'Graph': ops.ACTION,
            'Snapshot': ops.INSTANT_ACTION,
            'Foreach': ops.INSTANT_ACTION,
            'AsNumpy': ops.INSTANT_ACTION
        }

        op_type = operations_dict.get(name)

        if not op_type:
            raise Exception("Invalid operation \"{}\"".format(name))
        return op_type

    def is_action(self):
        """
        Checks if the current operation is an action.

        Returns:
            bool: True if the current operation is an action, False otherwise.
        """
        return self.op_type == Operation.Types.ACTION

    def is_transformation(self):
        """
        Checks if the current operation is a transformation.

        Returns:
            bool: True if the current operation is a transformation,
            False otherwise.
        """
        return self.op_type == Operation.Types.TRANSFORMATION

    def is_instant_action(self):
        """
        Checks if the current operation is an instant action.

        Returns:
            bool: True if the current operation is an instant action,
                False otherwise.
        """
        return self.op_type == Operation.Types.INSTANT_ACTION
