# Author: Enric Tejedor, Danilo Piparo CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import cppyy
import ROOT.pythonization as pyz

import functools
import importlib
import pkgutil

def pythonization(lazy = True):
    """
    Pythonizor decorator to be used in pythonization modules for pythonizations.
    These pythonizations functions are invoked upon usage of the class.
    Parameters
    ----------
    lazy : boolean
        If lazy is true, the class is pythonized upon first usage, otherwise
        upon import of the ROOT module.
    """
    def pythonization_impl(fn):
        """
        The real decorator. This structure is adopted to deal with parameters
        fn : function
            Function that implements some pythonization.
            The function must accept two parameters: the class
            to be pythonized and the name of that class.
        """
        if lazy:
            cppyy.py.add_pythonization(fn)
        else:
            fn()
    return pythonization_impl

# Trigger the addition of the pythonizations
for _, module_name, _ in  pkgutil.walk_packages(pyz.__path__):
    module = importlib.import_module(pyz.__name__ + '.' + module_name)

# Redirect ROOT to cppyy.gbl
import sys
sys.modules['ROOT'] = cppyy.gbl
