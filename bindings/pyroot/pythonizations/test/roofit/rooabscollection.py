import unittest

import ROOT


class TestRooAbsCollection(unittest.TestCase):
    """
    Test for RooAbsCollection pythonizations.
    """

    num_elems = 3
    rooabsarg_list = [ROOT.RooStringVar(str(i), "", "") for i in range(num_elems)]
    tlist = ROOT.TList()

    # Setup
    @classmethod
    def setUpClass(cls):
        for elem in cls.rooabsarg_list:
            cls.tlist.Add(elem)

    @classmethod
    def tearDownClass(cls):
        # Clear TList before Python list deletes the objects
        cls.tlist.Clear()

    # Parametrized tests for RooAbsCollection child classes
    def _test_len(self, collection_class):
        c = collection_class(self.tlist)
        self.assertEqual(len(c), self.num_elems)
        self.assertEqual(len(c), c.getSize())

    def _test_addowned(self, collection_class):
        coll = collection_class()
        if True:
            x = ROOT.RooRealVar("x", "x", -10, 10)
            coll.addOwned(x)
        self.assertTrue("x" in coll)

    def _test_iterator(self, collection_class):
        a = ROOT.RooRealVar("a", "", 0)
        b = ROOT.RooRealVar("b", "", 0)

        l = collection_class(a, b)

        it = iter(l)

        self.assertEqual(next(it).GetName(), "a")
        self.assertEqual(next(it).GetName(), "b")

        with self.assertRaises(StopIteration):
            next(it)

    # Tests
    def test_len_rooarglist(self):
        self._test_len(ROOT.RooArgList)

    def test_len_rooargset(self):
        self._test_len(ROOT.RooArgSet)

    def test_addowned_rooarglist(self):
        self._test_addowned(ROOT.RooArgList)

    def test_addowned_rooargset(self):
        self._test_addowned(ROOT.RooArgSet)

    def test_iterator_rooarglist(self):
        self._test_iterator(ROOT.RooArgList)

    def test_iterator_rooargset(self):
        self._test_iterator(ROOT.RooArgSet)


if __name__ == "__main__":
    unittest.main()
