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
        x_1 = ROOT.RooRealVar("x", "x", -10, 10)
        mu_1 = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
        sig_1 = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
        gauss_1 = ROOT.RooGaussian("gauss", "gaussian", x_1, mu_1, sig_1)
        x_2 = ROOT.RooRealVar("x", "x", -10, 10)
        mu_2 = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
        sig_2 = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
        gauss_2 = ROOT.RooGaussian("gauss", "gaussian", x_2, mu_2, sig_2)
        data = gauss_1.generate(ROOT.RooArgSet(x_1), 100)
        res_1 = gauss_1.fitTo(data, Range="sideband", Save=True)
        res_2 = gauss_2.fitTo(data, ROOT.RooFit.Range("sideband"), ROOT.RooFit.Save())
        self.assertTrue(res_1.isIdentical(res_2))

    def test_mixed_styles(self):
        # test that no error is causes if python style and cpp style
        # args are provided to fitto and that results are identical
        x_1 = ROOT.RooRealVar("x", "x", -10, 10)
        mu_1 = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
        sig_1 = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
        gauss_1 = ROOT.RooGaussian("gauss", "gaussian", x_1, mu_1, sig_1)
        x_2 = ROOT.RooRealVar("x", "x", -10, 10)
        mu_2 = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
        sig_2 = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
        gauss_2 = ROOT.RooGaussian("gauss", "gaussian", x_2, mu_2, sig_2)
        data = gauss_1.generate(ROOT.RooArgSet(x_1), 100)
        res_1 = gauss_1.fitTo(data, ROOT.RooFit.Range("sideband"), Save=True)
        res_2 = gauss_2.fitTo(data, ROOT.RooFit.Save(True), Range="sideband")
        self.assertTrue(res_1.isIdentical(res_2))


if __name__ == '__main__':
    unittest.main()
