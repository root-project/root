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

def _remove_pyz(self, o):
	# Parameters:
    # - self: collection
    # - o: object to remove from the collection
	res = self.Remove(o)

	if not res:
		raise ValueError('list.remove(x): x not in list')

def _extend_pyz(self, c):
	# Parameters:
    # - self: collection
    # - c: collection to extend self with
    lenc = c.GetEntries()
    it = TIter(c)
    for i in range(lenc):
    	self.Add(it.Next())

def _count_pyz(self, o):
	# Parameters:
    # - self: collection
    # - o: object to be counted in the collection
    # Returns:
    # - Number of occurrences of the object in the collection
	n = 0

	it = TIter(self)
	obj = it.Next()
	while obj:
		if obj == o:
			n += 1
		obj = it.Next()

	return n

# Python operators

def _add_pyz(self, c):
	# Parameters:
    # - self: first collection to be added
    # - c: second collection to be added
    # Returns:
    # - self + c
	res = self.__class__()
	_extend_pyz(res, self)
	_extend_pyz(res, c)
	return res

def _mul_pyz(self, n):
	# Parameters:
    # - self: collection to be multiplied
    # - n: factor to multiply the collection by
    # Returns:
    # - self * n
	res = self.__class__()
	for _ in range(n):
		_extend_pyz(res, self)
	return res

def _imul_pyz(self, n):
	# Parameters:
    # - self: collection to be multiplied (in place)
    # - n: factor to multiply the collection by
    # Returns:
    # - self *= n
	for _ in range(n - 1):
		_extend_pyz(self, self)
	return self

# Python iteration

def _begin_pyz(self):
	# Parameters:
    # - self: collection to be iterated
    # Returns:
    # - TIter iterator on collection
	return TIter(self)


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
        klass.remove = _remove_pyz
        klass.extend = _extend_pyz
        klass.count = _count_pyz

        # Define Python operators
        klass.__add__ = _add_pyz
        klass.__mul__ = _mul_pyz
        klass.__rmul__ = _mul_pyz
        klass.__imul__ = _imul_pyz

        # Make TCollections iterable.
        # In Pythonize.cxx, cppyy injects an `__iter__` method into
        # any class that has a `begin` and an `end` methods.
        # This is the case of TCollection and its subclasses, where
        # cppyy associates `__iter__` with a function that returns the
        # result of calling `begin`, i.e. a TIter that already points
        # to the first element of the collection.
        # That setting breaks the iteration on TCollections, since the
        # first time `Next` is called on the iterator it will return
        # the second element of the collection (if any), thus skipping
        # the first one.
        # By pythonising `begin` here, we make sure the iterator returned
        # points to nowhere, and it will return the first element of the
        # collection on the first invocation of `Next`.
        klass.begin = _begin_pyz

    return True
