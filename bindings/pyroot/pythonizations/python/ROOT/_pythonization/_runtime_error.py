# Author: Vincenzo Eduardo Padulano CERN  10/2023

################################################################################
# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization

# TODO: vpadulan
# This module enables pickling/unpickling of the std::runtime_error Python proxy
# defined in cppyy (via CPPExcInstance). The same logic should be implemented
# in the CPython extension to be more generic.

def _create_error(what):
    """
    Creates a new cppyy std::runtime_error proxy with the original message.
    """
    import cppyy
    return cppyy.gbl.std.runtime_error(what)

def _reduce_error(self):
    """
    Establish the strategy to recreate a std::runtime_error when unpickling.

    Creates a new error from the original message. Need to use a separate free
    function instead of an inline lambda to help pickle.
    """
    return _create_error, (self.what(), )

@pythonization("runtime_error", ns="std")
def pythonize_runtime_error(klass):
    """Add serialization capabilities to std::runtime_error."""
    klass.__reduce__ = _reduce_error
