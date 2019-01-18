# Author: Enric Tejedor CERN  01/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from cppyy.gbl import TIter

from ._generic import add_len


# Python-list-like methods

def remove_pyz(self, o):
	# Parameters:
    # self: collection
    # o: object to remove from the collection
	res = self.Remove(o)

	if not res:
		raise ValueError('list.remove(x): x not in list')

def extend_pyz(self, c):
	# Parameters:
    # self: collection
    # c: collection to extend self with
	it = TIter(c)
	o = it.Next()
	while o:
		self.Add(o)
		o = it.Next()

def count_pyz(self, o):
	# Parameters:
    # self: collection
    # o: object to be counted in the collection
	n = 0

	it = TIter(self)
	obj = it.Next()
	while obj:
		if obj == o:
			n += 1
		obj = it.Next()

	return n


@pythonization()
def pythonize_tcollection(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == 'TCollection':
        # Support `len(c)` as `c.GetEntries()`
        add_len(klass, 'GetEntries')

        # Add Python lists methods
        klass.append = klass.Add
        klass.remove = remove_pyz
        klass.extend = extend_pyz
        klass.count = count_pyz
