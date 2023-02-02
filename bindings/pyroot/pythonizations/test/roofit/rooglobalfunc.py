import unittest

import ROOT


class TestRooGlobalFunc(unittest.TestCase):
    """
    Test for RooGlobalFunc pythonizations.
    """

    def test_color_codes(self):
        """Test that the color code pythonizations in the functions like
        RooFit.LineColor are working as they should.
        """

        def code(color):
            """Get the color code that will be obtained by a given argument
            passed to RooFit.LineColor.
            """
            return ROOT.RooFit.LineColor(color).getInt(0)

        # Check that general string to enum pythonization works
        self.assertEqual(code(ROOT.kRed), code("kRed"))

        # Check that matplotlib-style color strings work
        self.assertEqual(code(ROOT.kRed), code("r"))

        # Check that postfix operations applied to ROOT color codes work
        self.assertEqual(code(ROOT.kRed + 1), code("kRed+1"))

    def test_roodataset_link(self):
        """Test that the RooFit.Link() command argument works as expected in
        the RooDataSet constructor.
        Inspired by the reproducer code in GitHub issue #11469.
        """
        x = ROOT.RooRealVar("x", "", 0, 1)
        g = ROOT.RooGaussian("g", "", x, ROOT.RooFit.RooConst(0.5), ROOT.RooFit.RooConst(0.2))

        n_events = 1000

        data = g.generate({x}, NumEvents=n_events)

        sample = ROOT.RooCategory("cat", "cat")
        sample.defineType("cat_0")

        data_2 = ROOT.RooDataSet("data_2", "data_2", {x}, Index=sample, Link={"cat_0": data})

        self.assertEqual(data_2.numEntries(), n_events)

    def test_minimizer(self):
        """C++ object returned by RooFit::Minimizer should not be double deleted"""
        # ROOT-9516
        minimizer = ROOT.RooFit.Minimizer("Minuit2", "migrad")


if __name__ == "__main__":
    unittest.main()
