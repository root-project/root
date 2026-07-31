import unittest

import ROOT
from ROOT import TString


class TStringLen(unittest.TestCase):
    """
    Test for the pythonization that provides the length of a
    TString instance `s` via `len(s)`.
    """

    # Tests
    def test_len(self):
        s = 'test'
        ts = ROOT.TString(s)
        self.assertEqual(len(ts), len(s))


class TStringStrRepr(unittest.TestCase):
    """
    Test for the pythonizations that provide a string representation
    for instances of TString (__str__, __repr__).
    """

    # Tests
    def test_str(self):
        s = 'test'
        ts = ROOT.TString(s)
        self.assertEqual(str(ts), s)

    def test_repr(self):
        s = 'test'
        ts = ROOT.TString(s)
        self.assertEqual(repr(ts), repr(s))


class TStringComparisonOps(unittest.TestCase):
    """
    Test for the comparison operators of TString:
    __eq__, __ne__, __lt__, __le__, __gt__, __ge__.
    """

    num_elems = 3
    test_str1 = 'test1'
    test_str2 = 'test2'

    # Tests
    def test_eq(self):
        ts1 = TString(self.test_str1)
        ts2 = TString(self.test_str1)
        ts3 = TString(self.test_str2)

        # Comparison between TStrings
        self.assertTrue(ts1 == ts2)
        self.assertFalse(ts1 == ts3)

        # Comparison with Python string
        self.assertTrue(ts1 == self.test_str1)
        self.assertFalse(ts1 == self.test_str2)

        # Comparison with non-string
        self.assertFalse(ts1 == 1)

    def test_ne(self):
        ts1 = TString(self.test_str1)
        ts2 = TString(self.test_str1)
        ts3 = TString(self.test_str2)

        # Comparison between TStrings
        self.assertFalse(ts1 != ts2)
        self.assertTrue(ts1 != ts3)

        # Comparison with Python string
        self.assertFalse(ts1 != self.test_str1)
        self.assertTrue(ts1 != self.test_str2)

        # Comparison with non-string
        self.assertTrue(ts1 != 1)

    def test_lt(self):
        ts1 = TString(self.test_str1)
        ts2 = TString(self.test_str2)

        # Comparison between TStrings
        self.assertTrue(ts1 < ts2)
        self.assertFalse(ts2 < ts1)

        # Comparison with Python string
        self.assertTrue(ts1 < self.test_str2)
        self.assertFalse(ts2 < self.test_str1)

    def test_le(self):
        ts1 = TString(self.test_str1)
        ts2 = TString(self.test_str1)
        ts3 = TString(self.test_str2)

        # Comparison between TStrings
        self.assertTrue(ts1 <= ts2)
        self.assertTrue(ts1 <= ts3)
        self.assertFalse(ts3 <= ts1)

        # Comparison with Python string
        self.assertTrue(ts1 <= self.test_str1)
        self.assertTrue(ts1 <= self.test_str2)
        self.assertFalse(ts3 <= self.test_str1)

    def test_gt(self):
        ts1 = TString(self.test_str1)
        ts2 = TString(self.test_str2)

        # Comparison between TStrings
        self.assertFalse(ts1 > ts2)
        self.assertTrue(ts2 > ts1)

        # Comparison with Python string
        self.assertFalse(ts1 > self.test_str2)
        self.assertTrue(ts2 > self.test_str1)

    def test_ge(self):
        ts1 = TString(self.test_str1)
        ts2 = TString(self.test_str1)
        ts3 = TString(self.test_str2)

        # Comparison between TStrings
        self.assertTrue(ts1 >= ts2)
        self.assertFalse(ts1 >= ts3)
        self.assertTrue(ts3 >= ts1)

        # Comparison with Python string
        self.assertTrue(ts1 >= self.test_str1)
        self.assertFalse(ts1 >= self.test_str2)
        self.assertTrue(ts3 >= self.test_str1)

    def test_list_sort(self):
        l1 = [ TString(str(i)) for i in range(self.num_elems) ]
        l2 = list(reversed(l1))

        self.assertNotEqual(l1, l2)

        # Test that comparison operators enable list sorting
        l2.sort()

        self.assertEqual(l1, l2)


class TStringConverter(unittest.TestCase):
    """
    Tests for passing a Python string to a C++ function that expects a TString.

    This feature is not implemented by a PyROOT pythonization, but by a converter
    that was added to Cppyy to create a TString out of a Python string.
    """

    test_str = "test"

    # Helpers
    def check_type_conversion(self):
        s = ROOT.TString(self.test_str)

        # Works with TString...
        self.assertEqual(ROOT.myfun(s), self.test_str)
        # ... and Python string
        self.assertEqual(ROOT.myfun(self.test_str), self.test_str)

    # Tests
    def test_by_value(self):
        ROOT.gInterpreter.Declare("""
        const char* myfun(TString s) { return s.Data(); }
        """)

        self.check_type_conversion()

    def test_by_reference(self):
        ROOT.gInterpreter.Declare("""
        const char* myfun(TString &s) { return s.Data(); }
        """)

        self.check_type_conversion()

    def test_by_const_reference(self):
        ROOT.gInterpreter.Declare("""
        const char* myfun(const TString &s) { return s.Data(); }
        """)

        self.check_type_conversion()


if __name__ == '__main__':
    unittest.main()
