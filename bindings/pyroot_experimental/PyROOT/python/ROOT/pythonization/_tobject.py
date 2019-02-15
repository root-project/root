# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
import cppyy

# Searching

def _contains(self, o):
	# Relies on TObject::FindObject
    # Parameters:
    # - self: object where to search
    # - o: object to be searched in self
    # Returns:
    # - True if self contains o
    return bool(self.FindObject(o))


@pythonization(lazy = False)
def pythonize_tobject():
	klass = cppyy.gbl.TObject

	# Allow 'obj in container' syntax for searching
	klass.__contains__ = _contains

	return True
