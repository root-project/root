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
from __future__ import annotations

import pkgutil
import types
import importlib


def build_backends_submodules(parentmodule):
    """
    Helper function to create the submodules of the backends.
    """
    for _, module_name, is_pkg in pkgutil.walk_packages(__path__):

        if is_pkg:
            # The actual python package with the backend implementation
            actual = importlib.import_module(__name__ + "." + module_name)
            # A dummy module to inject in the parent module
            fullmodulename = "ROOT.RDF.Experimental.Distributed." + module_name
            dummy = types.ModuleType(fullmodulename)

            # PEP302 attributes
            dummy.__file__ = "<module ROOT.RDF.Experimental.Distributed>"
            # dummy.__name__ is the constructor argument
            dummy.__path__ = []  # this makes it a package
            # dummy.__loader__ is not defined
            dummy.__package__ = parentmodule

            # Attached functions
            dummy.RDataFrame = actual.RDataFrame

            setattr(parentmodule, module_name, dummy)

    return parentmodule
