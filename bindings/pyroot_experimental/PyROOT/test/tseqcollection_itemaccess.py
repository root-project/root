import unittest

import ROOT
from libcppyy import SetOwnership


class TSeqCollectionItemAccess(unittest.TestCase):
    """
    Test for the item access methods added to TSeqCollection (and subclasses):
    __getitem__, __setitem__, __delitem__.
    Both the index (l[i]) and slice (l[i:j]) syntaxes are tested.
    """

    num_elems = 3

    # Helpers
    def create_tseqcollection(self):
        sc = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            SetOwnership(o, False)
            sc.Add(o)

        return sc

    # Tests
    def test_getitem(self):
        sc = self.create_tseqcollection()

        # Get elements in collection
        it = ROOT.TIter(sc)
        for i in range(self.num_elems):
            self.assertEqual(it.Next(), sc[i])

        # Check invalid index case
        with self.assertRaises(IndexError):
            sc[self.num_elems]

    def test_setitem(self):
        sc = self.create_tseqcollection()
        l = []
        
        # Set items
        for i in range(self.num_elems):
            o = ROOT.TObject()
            sc[i] = o
            l.append(o)

        # Check previously set items
        it = ROOT.TIter(sc)
        for i in range(self.num_elems):
            self.assertEqual(it.Next(), l[i])

        # Check invalid index case
        with self.assertRaises(IndexError):
            sc[self.num_elems] = ROOT.TObject()

    def test_delitem(self):
        sc = self.create_tseqcollection()

        self.assertEqual(sc.GetEntries(), self.num_elems)

        # Delete all elements
        for _ in range(self.num_elems):
            del sc[0]

        self.assertEqual(sc.GetEntries(), 0)

        sc = ROOT.TList()
        o1 = ROOT.TObject()
        o2 = ROOT.TObject()
        sc.Add(o1)
        sc.Add(o2)
        sc.Add(o1)

        # Delete o2
        del sc[1]

        # Only o1s should be there
        self.assertEqual(sc.GetEntries(), 2)

        it = ROOT.TIter(sc)
        for _ in range(2):
            self.assertEqual(it.Next(), o1)

        # Check invalid index case
        with self.assertRaises(IndexError):
            del sc[2]


if __name__ == '__main__':
    unittest.main()
