import unittest

import ROOT

class TestRooDataSet(unittest.TestCase):
    def test_createHistogram_decls(self):
        """RooDataSet::createHistogram overloads obtained with using decls."""

        import ROOT

        x = ROOT.RooRealVar("x", "x", -10, 10)
        mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
        sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
        gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

        data = gauss.generate(ROOT.RooArgSet(x), 10000)  # ROOT.RooDataSet
        h1d = data.createHistogram("myname", x)


if __name__ == "__main__":
    unittest.main()
