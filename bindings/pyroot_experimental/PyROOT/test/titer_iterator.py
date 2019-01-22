import unittest

import ROOT
from libcppyy import SetOwnership


class TIterIterator(unittest.TestCase):
    """
    Test for the pythonization that allows instances of TIter to
    behave as Python iterators.
    """

    num_elems = 3

    # Helpers
    def create_tcollection(self):
        c = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            SetOwnership(o, False)
            c.Add(o)

        return c

    # Tests
    def test_iterator(self):
        c = self.create_tcollection()

        itc1 = ROOT.TIter(c)
        itc2 = ROOT.TIter(c)
        for _ in range(c.GetEntries()):
            self.assertEqual(next(itc1), itc2.Next())
