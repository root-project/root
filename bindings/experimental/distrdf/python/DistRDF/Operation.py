#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from typing import Dict, Union


class Operation:
    """An operation attached to a distributed RDataFrame graph node."""

    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = list(args)
        self.kwargs = kwargs


class Action(Operation):
    """An action attached to a distributed RDataFrame graph node."""
    pass


class VariationsFor(Action):
    """
    DistRDF.VariationsFor creates a 'VariationsNode' node in the distributed
    RDataFrame graph. This acts as an action node.
    """
    pass


class InstantAction(Operation):
    """An instant action attached to a distributed RDataFrame graph node."""
    pass


class AsNumpy(InstantAction):
    """An 'AsNumpy' instant action attached to a distributed RDataFrame graph node."""
    pass


class Snapshot(InstantAction):
    """A 'Snapshot' instant action attached to a distributed RDataFrame graph node."""
    pass


class Transformation(Operation):
    """A trasformation attached to a distributed RDataFrame graph node."""
    pass


SUPPORTED_OPERATIONS: Dict[str, Union[Action, InstantAction, Transformation]] = {
    "AsNumpy": AsNumpy,
    "Count": Action,
    "Define": Transformation,
    "DefinePerSample": Transformation,
    "Filter": Transformation,
    "Graph": Action,
    "Histo1D": Action,
    "Histo2D": Action,
    "Histo3D": Action,
    "HistoND": Action,
    "Max": Action,
    "Mean": Action,
    "Min": Action,
    "Profile1D": Action,
    "Profile2D": Action,
    "Profile3D": Action,
    "Redefine": Transformation,
    "Snapshot": Snapshot,
    "Sum": Action,
    "VariationsFor": VariationsFor,
    "Vary": Transformation
}


def create_op(name: str, *args, **kwargs) -> Union[Action, InstantAction, Transformation]:
    try:
        return SUPPORTED_OPERATIONS[name](name, *args, **kwargs)
    except KeyError as e:
        raise ValueError(f"Operation '{name}' is either invalid or not supported in distributed mode. "
                         "See the documentation for a list of supported operations.") from e
