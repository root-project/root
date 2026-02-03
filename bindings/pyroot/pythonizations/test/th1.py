import unittest

import ROOT


class TH1Operators(unittest.TestCase):
    """
    Test for the __imul__ operator of TH1 and subclasses, which
    multiplies the histogram by a constant.
    """

    # Tests
    def test_imul(self):
        nbins = 64
        h = ROOT.TH1F("testHist", "", nbins, -4, 4)
        h.FillRandom("gaus")

        initial_bins = [h.GetBinContent(i) for i in range(nbins)]
        c = 2

        # Multiply in place
        h *= c

        # Check new value of bins
        for i in range(nbins):
            self.assertEqual(h.GetBinContent(i), initial_bins[i] * c)


class TH1IMT(unittest.TestCase):
    """
    Test a deadlock when IMT is used in conjunction with a fit function in Python.
    Since TH1.Fit held the GIL, the fit function could never be evaluated
    """

    @classmethod
    def setUpClass(cls):
        ROOT.ROOT.EnableImplicitMT(4)

    @classmethod
    def tearDownClass(cls):
        ROOT.ROOT.DisableImplicitMT()

    def test_fit_python_function(self):
        xmin = 0
        xmax = 1

        h1 = ROOT.TH1F("h1", "", 20, xmin, xmax)
        h1.FillRandom("gaus", 1000)

        def func(x, pars):
            return pars[0] + pars[1] * x[0]

        my_func = ROOT.TF1("f1", func, xmin, xmax, npar=2, ndim=1)

        my_func.SetParNames(
            "A",
            "B",
        )
        my_func.SetParameter(0, 1)
        my_func.SetParameter(1, -1)

        r = h1.Fit(my_func, "SE0Q", "", xmin, xmax)

        self.assertFalse(r.IsEmpty())
        self.assertTrue(r.IsValid())
        self.assertGreater(r.Parameter(0), 0)


if __name__ == "__main__":
    unittest.main()
