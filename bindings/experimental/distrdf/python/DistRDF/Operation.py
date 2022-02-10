#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from enum import Enum, auto


class OpTypes(Enum):
    ACTION = auto()
    INSTANT_ACTION = auto()
    TRANSFORMATION = auto()


DISTRDF_SUPPORTED_OPERATIONS = {
    "AsNumpy": OpTypes.INSTANT_ACTION,
    "Count": OpTypes.ACTION,
    "Define": OpTypes.TRANSFORMATION,
    "DefinePerSample": OpTypes.TRANSFORMATION,
    "Filter": OpTypes.TRANSFORMATION,
    "Graph": OpTypes.ACTION,
    "Histo1D": OpTypes.ACTION,
    "Histo2D": OpTypes.ACTION,
    "Histo3D": OpTypes.ACTION,
    "HistoND": OpTypes.ACTION,
    "Max": OpTypes.ACTION,
    "Mean": OpTypes.ACTION,
    "Min": OpTypes.ACTION,
    "Profile1D": OpTypes.ACTION,
    "Profile2D": OpTypes.ACTION,
    "Profile3D": OpTypes.ACTION,
    "Redefine": OpTypes.TRANSFORMATION,
    "Snapshot": OpTypes.INSTANT_ACTION,
    "Sum": OpTypes.ACTION,
}


class Operation(object):
    """
    A Generic representation of an operation. The operation could be a
    transformation, an action or an instant action.

    Attributes:

        name (str): Name of the current operation.

        args (list): Variable length argument list for the current operation.

        kwargs (dict): Arbitrary keyword arguments for the current operation.

        op_type: The type or category of the current operation
            (`ACTION`, `TRANSFORMATION` or `INSTANT_ACTION`).

    """

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
        try:
            self.op_type = DISTRDF_SUPPORTED_OPERATIONS[name]
        except KeyError as e:
            raise RuntimeError(f"Operation '{name}' is either invalid or not supported in distributed mode. "
                               "See the documentation for a list of supported operations.") from e

    def is_action(self):
        """
        Checks if the current operation is an action.

        Returns:
            bool: True if the current operation is an action, False otherwise.
        """
        return self.op_type == OpTypes.ACTION

    def is_transformation(self):
        """
        Checks if the current operation is a transformation.

        Returns:
            bool: True if the current operation is a transformation,
            False otherwise.
        """
        return self.op_type == OpTypes.TRANSFORMATION

    def is_instant_action(self):
        """
        Checks if the current operation is an instant action.

        Returns:
            bool: True if the current operation is an instant action,
                False otherwise.
        """
        return self.op_type == OpTypes.INSTANT_ACTION
