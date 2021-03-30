# Authors:
# * Harshal Shende  03/2021
# * Hinnerk C. Schmidt 02/2021
# * Jonas Rembser 03/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import cppyy

from ROOT import pythonization

from ._rooabsdata import RooAbsData
from ._rooabspdf import RooAbsPdf
from ._rooabsreal import RooAbsReal
from ._rooworkspace import RooWorkspace


# list of python classes that are used to pythonize RooFit classes
python_classes = [RooAbsData, RooAbsPdf, RooAbsReal, RooWorkspace]

# create a dictionary for convenient access to python classes
python_classes_dict = dict()
for python_class in python_classes:
    python_classes_dict[python_class.__name__] = python_class


def ismagicfunc(name):
    return name.startswith("__") and name.endswith("__")


@pythonization()
def pythonize_roofit(klass, name):
    # Parameters:
    # klass: class to pythonize
    # name: string containing the name of the class

    if not name.startswith("Roo"):
        return

    if not name in python_classes_dict:
        return

    python_klass = python_classes_dict[name]

    # list of functions to pythonize, which are assumed to be all functions in
    # the pyton classes that are not magic functions
    func_names = [f for f in dir(python_klass) if not ismagicfunc(f)]

    for func_name in func_names:

        # if the RooFit class already has a function with the same name as our
        # pythonization, we rename it and prefix it with an underscore
        if hasattr(klass, func_name):
            # new name for original function
            func_name_orig = "_" + func_name

            setattr(klass, func_name_orig, getattr(klass, func_name))

        setattr(klass, func_name, getattr(python_klass, func_name))

    return
