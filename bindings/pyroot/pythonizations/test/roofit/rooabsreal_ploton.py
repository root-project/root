import unittest

import ROOT


class RooAbsRealPlotOn(unittest.TestCase):
    """
    Test for the PlotOn callable.
    """

    x = ROOT.RooRealVar("x", "x", -10, 10)
    mu = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
    sig = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gaussian", x, mu, sig)
    data = gauss.generate(ROOT.RooArgSet(x), 100)
    xframe = x.frame(ROOT.RooFit.Title("Gaussian pdf"))

    def test_frame(self):
        # test that kwargs can be passed
        # and lead to correct result
        r1 = self.gauss.plotOn(self.xframe, ROOT.RooFit.LineColor(ROOT.kRed))
        r2 = self.gauss.plotOn(self.xframe, LineColor = ROOT.kRed)

    def test_wrong_kwargs(self):
        # test that AttributeError is raised
        # if keyword does not correspong to CmdArg
        self.assertRaises(AttributeError, self.gauss.plotOn, self.xframe, ThisIsNotACmgArg=True)

    def test_binning(self):
        # test that fitting with keyword arguments leads to the same result
        # as doing the same plot with passed ROOT objects
        dtframe = self.x.frame(ROOT.RooFit.Range(-5, 5), ROOT.RooFit.Title("dt distribution with custom binning"))
        binning = ROOT.RooBinning(20, -5, 5)
        r1 = self.data.plotOn(dtframe, ROOT.RooFit.Binning(binning))
        r2 = self.data.plotOn(dtframe, Binning = binning)

    def test_data(self):
        # test that no error is causes if python style and cpp style
        # args are provided to plotOn and that results are identical
        frame = self.x.frame(ROOT.RooFit.Name("xframe"), ROOT.RooFit.Title("Red Curve"), ROOT.RooFit.Bins(20))
        res1_d1 = self.data.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
        res2_d1 = self.data.plotOn(frame, DataError = ROOT.RooAbsData.SumW2 )


if __name__ == '__main__':
    unittest.main()
