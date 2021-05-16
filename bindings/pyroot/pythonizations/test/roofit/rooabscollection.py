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

    def _test_contains(self, collection_class):
        var0 = ROOT.RooRealVar("var0", "var0", 0)
        var1 = ROOT.RooRealVar("var1", "var1", 1)

        # The next variable has a duplicate name on purpose the check if it's
        # really the name that is used as the key in RooAbsCollections.
        var2 = ROOT.RooRealVar("var0", "var0", 0)

        coll = collection_class(var0)

        # expected behaviour
        self.assertTrue(var0 in coll)
        self.assertTrue("var0" in coll)

        self.assertTrue(not var1 in coll)
        self.assertTrue(not "var1" in coll)

        self.assertTrue(var2 in coll)
        self.assertTrue(not "var2" in coll)

        # ensure consistency with RooAbsCollection::find
        variables = [var0, var1, var2]

        for i, vptr in enumerate(variables):
            vname = "var" + str(i)
            self.assertEqual(coll.find(vptr) == vptr, vptr in coll)
            self.assertEqual(coll.find(vname) == vptr, vname in coll)

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

    def test_contains_rooarglist(self):
        self._test_contains(ROOT.RooArgList)

    def test_contains_rooargset(self):
        self._test_contains(ROOT.RooArgSet)


if __name__ == "__main__":
    unittest.main()
