import unittest

import ROOT


class TestRooArgList(unittest.TestCase):
    """
    Tests for RooArgList.
    """

    # Tests
    def test_rooarglist_iterator(self):
        a = ROOT.RooRealVar("a", "", 0)
        b = ROOT.RooRealVar("b", "", 0)

        l = ROOT.RooArgList(a, b)

        it = iter(l)

        self.assertEqual(next(it).GetName(), "a")
        self.assertEqual(next(it).GetName(), "b")

        with self.assertRaises(StopIteration):
            next(it)


if __name__ == "__main__":
    unittest.main()
