# Author: Ivan Kabadzhov, Vincenzo Eduardo Padulano CERN  01/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization

@pythonization("RDFDescription", ns="ROOT::RDF")
def pythonize_rdfdescription(klass):
    """
    Parameters:
    klass: class to be pythonized
    """
    klass.__repr__ = klass.AsString
