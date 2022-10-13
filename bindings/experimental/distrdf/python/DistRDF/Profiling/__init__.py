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

import os

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

        # We check whether the environment variable is set by the user in the
        # local environment. For a local run of distributed RDataFrame, this
        # should be enough to also get jitting info from cling in the mapper
        # tasks. For distributed runs, the variable should be also propagated to the workers
        if os.environ.get("CLING_PROFILE", "0") != "1":
            raise RuntimeError(
                "The profiling feature is not active. Please set CLING_PROFILE=1 in your environment and make sure it "
                "is also propagated to the distributed nodes.")

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

    # Before running the mapper, check that this worker has the environment
    # variable set. Hopefully this was set in a way that when cling was loaded
    # for the first time on this workers, the variable was already visible.
    if os.environ.get("CLING_PROFILE", "0") != "1":
        raise RuntimeError(
            "The profiling feature is not active in the environment of the computing nodes. "
            "Please set CLING_PROFILE=1 on all nodes of the cluster.")

    # Import of DistRDF mapper is done only here
    from DistRDF.Backends.Base import distrdf_mapper
    return distrdf_mapper(*args, **kwargs)
