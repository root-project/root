# @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

class Operation(object):
    """
    A Generic representation of an operation. The operation could be a
    transformation, an action or an instant action.

    Attributes:
        ACTION (str): Action type string.

        INSTANT_ACTION (str): Instant action type string.

        TRANSFORMATION (str): Transformation type string.

        name (str): Name of the current operation.

        args (list): Variable length argument list for the current operation.

        kwargs (dict): Arbitrary keyword arguments for the current operation.

        op_type: The type or category of the current operation
            (`ACTION`, `TRANSFORMATION` or `INSTANT_ACTION`).

    """

    ACTION = "ACTION"
    INSTANT_ACTION = "INSTANT_ACTION"
    TRANSFORMATION = "TRANSFORMATION"

    def __init__(self, name, *args, **kwargs):
        """
        Creates a new `Operation` for the given name and arguments.

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
        """
        Classifies the given operation as action or transformation and
        returns the type.
        """

        operations_dict = {
            "Define": Operation.TRANSFORMATION,
            "DefinePerSample": Operation.TRANSFORMATION,
            "Filter": Operation.TRANSFORMATION,
            "Range": Operation.TRANSFORMATION,
            "Aggregate": Operation.ACTION,
            "Histo1D": Operation.ACTION,
            "Histo2D": Operation.ACTION,
            "Histo3D": Operation.ACTION,
            "HistoND": Operation.ACTION,
            "Profile1D": Operation.ACTION,
            "Profile2D": Operation.ACTION,
            "Profile3D": Operation.ACTION,
            "Count": Operation.ACTION,
            "Min": Operation.ACTION,
            "Max": Operation.ACTION,
            "Mean": Operation.ACTION,
            "Sum": Operation.ACTION,
            "Fill": Operation.ACTION,
            "Redefine": Operation.TRANSFORMATION,
            "Reduce": Operation.ACTION,
            "Report": Operation.ACTION,
            "Take": Operation.ACTION,
            "Graph": Operation.ACTION,
            "Snapshot": Operation.INSTANT_ACTION,
            "Foreach": Operation.INSTANT_ACTION,
            "AsNumpy": Operation.INSTANT_ACTION
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
        return self.op_type == Operation.ACTION

    def is_transformation(self):
        """
        Checks if the current operation is a transformation.

        Returns:
            bool: True if the current operation is a transformation,
            False otherwise.
        """
        return self.op_type == Operation.TRANSFORMATION

    def is_instant_action(self):
        """
        Checks if the current operation is an instant action.

        Returns:
            bool: True if the current operation is an instant action,
                False otherwise.
        """
        return self.op_type == Operation.INSTANT_ACTION
