# Author: Enric Tejedor CERN  01/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
import cppyy

def _next_pyz(self):
    # Parameters:
    # - self: iterator on a collection
    # Returns:
    # - next object in the collection, or raises StopIteration if none
	o = self.Next()
	if o:
		return o
	else:
		raise StopIteration()

# The TIter class does not go through the mechanism of lazy pythonisations of
# cppyy, since it is used before such mechanism is put in place. Therefore, we
# define here the pythonisations for TIter as immediate, i.e. executed upon
# import of the ROOT module
@pythonization(lazy = False)
def pythonize_titer():
    klass = cppyy.gbl.TIter

    # Make TIter a Python iterator.
    # This makes it possible to iterate over TCollections, since Cppyy
    # injects on them an `__iter__` method that returns a TIter.
    klass.__next__ = _next_pyz  # Py3
    klass.next     = _next_pyz  # Py2

    return True
