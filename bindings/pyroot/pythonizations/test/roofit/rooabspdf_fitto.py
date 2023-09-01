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

    x.setRange("sideband", -10, 0)

    def _reset_initial_values(self):
        # after every fit, we have to reset the initial values because
        # otherwise the fit result is not considered the same
        self.mu.setVal(1.0)
        self.sig.setVal(1.0)
        self.mu.setError(0.0)
        self.sig.setError(0.0)

    def test_save(self):
        # test that kwargs can be passed
        # and lead to correct result
        gauss = self.gauss
        data = self.data
        self.assertEqual(
            gauss.fitTo(data, Save=False, PrintLevel=-1),
            gauss.fitTo(data, ROOT.RooFit.Save(ROOT.kFALSE), ROOT.RooFit.PrintLevel(-1)),
        )
        self.assertTrue(bool(gauss.fitTo(data, Save=True, PrintLevel=-1)))
        self._reset_initial_values()

    def test_wrong_kwargs(self):
        # test that AttributeError is raised
        # if keyword does not correspong to CmdArg
        gauss = self.gauss
        data = self.data
        self.assertRaises(AttributeError, gauss.fitTo, data, ThisIsNotACmgArg=True)
        self._reset_initial_values()

    def test_identical_result(self):
        # test that fitting with keyword arguments leads to the same result
        # as doing the same fit with passed ROOT objects
        gauss = self.gauss
        data = self.data
        res_1 = gauss.fitTo(data, Range="sideband", Save=True, PrintLevel=-1)
        self._reset_initial_values()
        res_2 = gauss.fitTo(data, ROOT.RooFit.Range("sideband"), ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
        self._reset_initial_values()
        self.assertTrue(res_1.isIdentical(res_2))

    def test_mixed_styles(self):
        # test that no error is causes if python style and cpp style
        # args are provided to fitto and that results are identical
        gauss = self.gauss
        data = self.data
        res_1 = gauss.fitTo(data, ROOT.RooFit.Range("sideband"), Save=True, PrintLevel=-1)
        self._reset_initial_values()
        res_2 = gauss.fitTo(data, ROOT.RooFit.Save(True), Range="sideband", PrintLevel=-1)
        self._reset_initial_values()
        self.assertTrue(res_1.isIdentical(res_2))


if __name__ == "__main__":
    unittest.main()
