import unittest

import ROOT


class RooAbsPdfFitTo(unittest.TestCase):
    """
    Test for the FitTo callable.
    """

    x = ROOT.RooRealVar("x", "x", -10, 10)
    mu = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
    sig = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gaussian", x, mu, sig)
    data = gauss.generate(ROOT.RooArgSet(x), 100)

    def test_save(self):
        # test that kwargs can be passed
        # and lead to correct result
        self.assertEqual(self.gauss.fitTo(self.data, Save=False), self.gauss.fitTo(self.data, ROOT.RooFit.Save(ROOT.kFALSE)))
        self.assertTrue(bool(self.gauss.fitTo(self.data, Save=True)))

    def test_wrong_kwargs(self):
        # test that AttributeError is raised 
        # if keyword does not correspong to CmdArg
        self.assertRaises(AttributeError, self.gauss.fitTo, self.data, ThisIsNotACmgArg=True)

    def test_identical_result(self):
        # test that fitting with keyword arguments leads to the same result
        # as doing the same fit with passed ROOT objects
        x = ROOT.RooRealVar("x", "x", -10, 10)
        mu = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
        sig = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
        gauss2 = ROOT.RooGaussian("gauss", "gaussian", x, mu, sig)
        res1 = self.gauss.fitTo(self.data, Range="sideband", Save=True)
        res2 = gauss2.fitTo(self.data, ROOT.RooFit.Range("sideband"), ROOT.RooFit.Save())
        self.assertTrue(res1.isIdentical(res2))


if __name__ == '__main__':
    unittest.main()
