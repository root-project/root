#  @author Giulio Crognaletti
#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2022-08

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

# Inform cling to activate dumping of profiling information for jitted code.
# It is crucial that this instruction is executed before importing ROOT in the distributed workers.
from __future__ import annotations

# TODO: Each time the module DistRDF is imported, the CLING_PROFILE=1 variable is set
# modyfing the environment in both the client and the workers.
# This affects only the workers when the profling option is enabled, since in all other
# cases Cling is already initialized. However the environment is modified every time, and
# this can potentially give rise to unexpected side effects.
# The issue can be solved by adding the option to programmatically enable/disable 
# Cling's profiling feature after its initialization, moving this step inside the 
# data collection context manager for all Visualizations
from os import environ
environ["CLING_PROFILE"]="1"

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..DataFrame import RDataFrame

from .Flamegraph import FlameGraph

SUPPORTED_VISUALIZATIONS = {
    "flamegraph": FlameGraph
}

class ClingProfile():
    """
    Context manager to enable profiling of DistRDF applications.

    All DistRDF code executed within the ClingProfile context will collect performance metric data in the workers and
    merge them together in a single visualization (e.g. a flamegraph).
    """

    def __init__(self, rdf:RDataFrame, visualization:str="flamegraph", **options) -> None:
        
        vclass = SUPPORTED_VISUALIZATIONS.get(visualization, None)

        if not vclass:
            raise ValueError("Visualization {visualization} is not supported.")
        
        self.visualization = vclass(**options)
        self.rdf = rdf

    def __enter__(self):
        
        self.rdf._headnode._visualization = self.visualization
        self.rdf._headnode._activate_profiling = True

    def __exit__(self, *exc):
        
        self.rdf._headnode._activate_profiling = False
        self.rdf._headnode._visualization = None

def profilable_mapper(*args, **kwargs):
    """DistRDF mapper function wrapper"""

    # This wrapper is used to trick pickle, and avoid import of the dependencies of distrdf_mapper, which include ROOT.
    # By doing so, this submodule gets imported first, so the CLING_PROFILE=1 environment variable is set before cling
    # gets initialized, hence effectively enabling the profiling feature.

    # Import of DistRDF mapper is done only here
    from DistRDF.Backends.Base import distrdf_mapper
    return distrdf_mapper(*args, **kwargs)
