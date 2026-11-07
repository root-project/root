import unittest

import ROOT


class TCollectionLen(unittest.TestCase):
    """
    Test for the pythonization that allows to access the number of elements of a
    TCollection (or subclass) by calling `len` on it.
    """

    num_elems = 3
    tobject_list = [ ROOT.TObject() for _ in range(num_elems) ]

    # Helpers
    def add_elems_check_len(self, c):
        for elem in self.tobject_list:
            c.Add(elem)

        self.assertEqual(len(c), self.num_elems)
        self.assertEqual(len(c), c.GetEntries())

    # Tests
    def test_tlist(self):
        self.add_elems_check_len(ROOT.TList())

    def test_tobjarray(self):
        self.add_elems_check_len(ROOT.TObjArray())

    def test_thashtable(self):
        self.add_elems_check_len(ROOT.THashTable())


class TCollectionListMethods(unittest.TestCase):
    """
    Test for the Python-list-like methods added to TCollection (and subclasses):
    append, remove, extend, count
    """

    num_elems = 3

    _global_objects = []

    # Helpers
    def create_tcollection(self):
        c = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            c.Add(o)
            # To prevent deletion of the objects (TList is by default non-owning)
            self._global_objects.append(o)

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
        self.assertIs(o, itc.Next())

        # Clear before the added element might be garbage collected,
        # to avoid dangling pointer access.
        c.Clear()

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
            self.assertIs(itc1.Next(), itc2.Next())


class TCollectionOperators(unittest.TestCase):
    """
    Test for the Python operators defined in TCollection (and subclasses):
    __add__, __mul__, __rmul__, __imul__
    """

    num_elems = 3
    factor = 2

    # Helpers
    def create_tcollection(self):
        c = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            ROOT.SetOwnership(o, False)
            c.Add(o)

        return c

    def check_mul_result(self, c, cmul):
        lenc = c.GetEntries()

        self.assertEqual(cmul.GetEntries(), lenc * self.factor)

        itmul = ROOT.TIter(cmul)
        for _ in range(self.factor):
            itc = ROOT.TIter(c)
            for _ in range(lenc):
                oc = itc.Next()
                omul = itmul.Next()
                self.assertIs(oc, omul)

    # Tests
    def test_add(self):
        c1 = self.create_tcollection()
        c2 = self.create_tcollection()

        len1 = c1.GetEntries()
        len2 = c2.GetEntries()

        cadd = c1 + c2

        len_add = cadd.GetEntries()

        self.assertEqual(len_add, len1 + len2)

        # Compare with elements of first collection
        itc1 = ROOT.TIter(c1)
        itadd = ROOT.TIter(cadd)
        for _ in range(len1):
            oc1 = itc1.Next()
            oadd = itadd.Next()
            self.assertIs(oc1, oadd)

        # Compare with elements of second collection
        itc2 = ROOT.TIter(c2)
        for _ in range(len2):
            oc2 = itc2.Next()
            oadd = itadd.Next()
            self.assertIs(oc2, oadd)

    def test_mul(self):
        c = self.create_tcollection()

        cmul = c * self.factor

        self.check_mul_result(c, cmul)

    def test_rmul(self):
        c = self.create_tcollection()

        cmul = self.factor * c

        self.check_mul_result(c, cmul)

    def test_imul(self):
        c = self.create_tcollection()
        lenc = c.GetEntries()

        c *= self.factor

        self.assertEqual(c.GetEntries(), lenc * self.factor)

        it = ROOT.TIter(c)
        subc = []
        for _ in range(lenc):
            subc.append(it.Next())

        for _ in range(self.factor - 1):
            for o in subc:
                self.assertIs(o, it.Next())


class TCollectionIterable(unittest.TestCase):
    """
    Test for the pythonization that makes instances of TCollection subclasses
    iterable in Python.
    For example, this allows to do:
    `for elem in collection:`
         `...`
    """

    num_elems = 3

    # Helpers
    def create_tcollection(self):
        c = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            ROOT.SetOwnership(o, False)
            c.Add(o)

        return c

    # Tests
    def test_iterable(self):
        c = self.create_tcollection()

        itc = ROOT.TIter(c)
        for elem in c:
            self.assertIs(elem, itc.Next())


if __name__ == '__main__':
    unittest.main()
