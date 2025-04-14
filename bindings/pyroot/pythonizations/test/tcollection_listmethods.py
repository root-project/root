import unittest

import ROOT


class TCollectionListMethods(unittest.TestCase):
    """
    Test for the Python-list-like methods added to TCollection (and subclasses):
    append, remove, extend, count
    """

    num_elems = 3

    # Helpers
    def create_tcollection(self):
        c = ROOT.TList()
        pylist = []
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            ROOT.SetOwnership(o, False)
            c.Add(o)
            pylist.append(o)

        # To prevent memory leaks
        c._owned_objects = pylist
        return c

    # Tests
    def test_append(self):
        c = self.create_tcollection()

        o = ROOT.TObject()
        self.assertFalse(c.Contains(o))
        len1 = c.GetEntries()

        c.append(o)

        len2 = c.GetEntries()
        self.assertEqual(len1 + 1, len2)
        self.assertTrue(c.Contains(o))

        # Skip elements that were already there
        itc = ROOT.TIter(c)
        for _ in range(len1):
            itc.Next()

        # Check that `o` is indeed the last element
        self.assertEqual(o, itc.Next())

    def test_remove(self):
        c = ROOT.TList()

        o1 = ROOT.TObject()
        o2 = ROOT.TObject()

        c.Add(o1)
        c.Add(o2)
        c.Add(o1)

        self.assertTrue(c.Contains(o1))
        self.assertEqual(c.GetEntries(), 3)

        c.remove(o1)

        self.assertEqual(c.GetEntries(), 2)

        c.remove(o1)

        self.assertEqual(c.GetEntries(), 1)
        self.assertFalse(c.Contains(o1))

        with self.assertRaises(ValueError):
            c.remove(o1)

        # The TList destructor will call TList::Clear(), which triggers a
        # ROOT-internal garbage collection routine implemented in
        # TCollection::GarbageCollect(). This can segfault if Python already
        # garbage collected the objects in the TList itself, because then the
        # TList has dangling pointers. To avoid this, we call Clear() now,
        # before the reference count of the objects in the list goes to zero.
        c.Clear()

    def test_extend(self):
        c1 = self.create_tcollection()
        c2 = self.create_tcollection()

        len1 = c1.GetEntries()
        len2 = c2.GetEntries()

        c1.extend(c2)

        len1_final = c1.GetEntries()

        self.assertEqual(len1_final, len1 + len2)

        # Skip elements that were already there
        itc1 = ROOT.TIter(c1)
        for _ in range(len1):
            itc1.Next()

        # Compare with elements of second collection
        itc2 = ROOT.TIter(c2)
        for _ in range(len2):
            self.assertEqual(itc1.Next(), itc2.Next())

    def test_count(self):
        c = ROOT.TList()

        o1 = ROOT.TObject()
        o2 = ROOT.TObject()

        c.Add(o1)
        c.Add(o2)
        c.Add(o1)

        self.assertEqual(c.count(o1), 2)
        self.assertEqual(c.count(o2), 1)

        # The TList destructor will call TList::Clear(), which triggers a
        # ROOT-internal garbage collection routine implemented in
        # TCollection::GarbageCollect(). This can segfault if Python already
        # garbage collected the objects in the TList itself, because then the
        # TList has dangling pointers. To avoid this, we call Clear() now,
        # before the reference count of the objects in the list goes to zero.
        c.Clear()


if __name__ == '__main__':
    unittest.main()

