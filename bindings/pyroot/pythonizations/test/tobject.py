import unittest

import ROOT
from ROOT import TUrl


class TObjectContains(unittest.TestCase):
    """
    Test for the __contains__ pythonization of TObject and subclasses.
    Such pythonization relies on TObject::FindObject, which is redefined
    in some of its subclasses, such as TCollection.
    Thanks to this pythonization, we can use the syntax `obj in col`
    to know if col contains obj.
    """

    num_elems = 3

    # Helpers
    def create_tlist(self):
        l = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            ROOT.SetOwnership(o, False)
            l.Add(o)

        return l

    # Tests
    def test_contains(self):
        l = self.create_tlist()

        for elem in l:
            self.assertTrue(elem in l)
            # Make sure it does not work just because of __iter__
            self.assertTrue(l.__contains__(elem))

        o = ROOT.TObject()
        self.assertFalse(o in l)
        self.assertFalse(l.__contains__(o))


class TObjectComparisonOps(unittest.TestCase):
    """
    Test for the comparison operators of TObject and subclasses:
    __eq__, __ne__, __lt__, __le__, __gt__, __ge__.
    The ordering pythonizations rely on TObject::Compare, which can be
    reimplemented in subclasses. There is no __eq__/__ne__ pythonization:
    equality follows C++ semantics, so comparing objects of a class without
    a C++ equality operator raises a TypeError.
    """

    num_elems = 3

    # Tests
    def test_eq(self):
        o = ROOT.TObject()

        # TObject has no C++ operator==, so the comparison is not supported.
        # To compare by address, TObject::IsEqual can be used explicitly.
        with self.assertRaises(TypeError):
            o == o
        self.assertTrue(o.IsEqual(o))

        # Test comparison with no TObject
        self.assertFalse(o == 1)

        # Test comparison with None
        self.assertFalse(o == None)

    def test_ne(self):
        o = ROOT.TObject()

        # TObject has no C++ operator!=, so the comparison is not supported.
        # To compare by address, TObject::IsEqual can be used explicitly.
        with self.assertRaises(TypeError):
            o != o
        self.assertFalse(not o.IsEqual(o))

        # Test comparison with no TObject
        self.assertTrue(o != 1)

        # Test comparison with None
        self.assertTrue(o != None)

    def test_nullptr_eq_none_raises(self):
        import cppyy

        x = cppyy.bind_object(cppyy.nullptr, "TObject")

        # Comparing a nullptr to None must raise TypeError in ROOT >= 6.40
        # This is important to check, because if we don't raise an error, the
        # result might not be equivalent to the confusing behavior in previous
        # ROOT versions, which would be a silent behavior change.
        # See https://github.com/root-project/root/issues/20283
        with self.assertRaises(TypeError):
            _ = x == None

        with self.assertRaises(TypeError):
            _ = x != None

    def test_lt(self):
        a = TUrl("a")
        b = TUrl("b")

        # TUrl::Compare compares URL strings
        self.assertTrue(a < b)
        self.assertFalse(b < a)

        # Test comparison with no TObject
        self.assertEqual(a.__lt__(1), NotImplemented)

    def test_le(self):
        a1 = TUrl("a")
        a2 = TUrl("a")
        b  = TUrl("b")

        # TUrl::Compare compares URL strings
        self.assertTrue(a1 <= a2)
        self.assertTrue(a2 <= a1)
        self.assertTrue(a1 <= b)
        self.assertFalse(b <= a1)

        # Test comparison with no TObject
        self.assertEqual(a1.__le__(1), NotImplemented)

    def test_gt(self):
        a = TUrl("a")
        b = TUrl("b")

        # TUrl::Compare compares URL strings
        self.assertFalse(a > b)
        self.assertTrue(b > a)

        # Test comparison with no TObject
        self.assertEqual(a.__gt__(1), NotImplemented)

    def test_ge(self):
        a1 = TUrl("a")
        a2 = TUrl("a")
        b  = TUrl("b")

        # TUrl::Compare compares URL strings
        self.assertTrue(a1 >= a2)
        self.assertTrue(a2 >= a1)
        self.assertTrue(b >= a1)
        self.assertFalse(a1 >= b)

        # Test comparison with no TObject
        self.assertEqual(a1.__ge__(1), NotImplemented)

    def test_list_sort(self):
        l1 = [ ROOT.TUrl(str(i)) for i in range(self.num_elems) ]
        l2 = list(reversed(l1))

        for i in range(self.num_elems):
            self.assertIs(l1[i], l2[self.num_elems - 1 - i])

        # Test that comparison operators enable list sorting
        l2.sort()

        for e1, e2 in zip(l1, l2):
            self.assertIs(e1, e2)


if __name__ == '__main__':
    unittest.main()
