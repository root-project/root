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

import ROOT


class Operation:
    """An operation attached to a distributed RDataFrame graph node."""

    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = list(args)
        self.kwargs = kwargs


class Action(Operation):
    """An action attached to a distributed RDataFrame graph node."""
    pass


class Histo(Action):
    """
    Any type of Histo*D action.

    This distinct class is needed to check that the user passes a model for the
    histogram action. Merging histograms coming from different distributed tasks
    is not possible unless the model was previously specified, since the binning
    of the different histograms would be incompatible.
    """

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        # The histogram model can be passed as the keyword argument 'model'. All
        # Histo*D specializations have the same name for this argument. If it is
        # present, we know the execution can proceed safely.
        if not "model" in self.kwargs:
            # If the keyword argument was not passed, we need to check the first
            # positional argument. In all Histo*D overload where it is present,
            # it is always the first argument.
            if not isinstance(self.args[0],
                              (tuple, ROOT.RDF.TH1DModel, ROOT.RDF.TH2DModel, ROOT.RDF.TH3DModel, ROOT.RDF.THnDModel)):
                message = (
                    "Creating a histogram without a model is not supported in distributed mode. Please make sure to "
                    "specify the histogram model when rerunning the distributed RDataFrame application. For example:\n\n"
                    "\tHisto1D('mycolumn') --> Histo1D(('myhist', 'myhist', 100, 0, 10), 'mycolumn')\n\n"
                    "See the RDataFrame documentation for more details.")
                raise ValueError(message)


class VariationsFor(Action):
    """
    DistRDF.VariationsFor creates a specific node in the distributed
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
    "Histo1D": Histo,
    "Histo2D": Histo,
    "Histo3D": Histo,
    "HistoND": Histo,
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
