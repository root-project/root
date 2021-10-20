#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-10


################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization

@pythonization()
def pythonize_rresultptr(klass, name):
    """
    Pythonization for RResultPtr that always releases the GIL when triggering
    the RDF computation graph.
    """

    if name.startswith("ROOT::RDF::RResultPtr<"):
        klass.GetValue.__release_gil__ = True

    return True
