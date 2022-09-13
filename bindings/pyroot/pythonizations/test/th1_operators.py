import unittest

import ROOT


class TH1Operators(unittest.TestCase):
    """
    Test for the __imul__ operator of TH1 and subclasses, which
    multiplies the histogram by a constant.
    """

    # Tests
    def test_numpy(self):
        import numpy as np

        x = np.random.normal(loc=0.0, scale=1.0, size=10000)
        w = np.ones(10000)
        h1 = ROOT.TH1.FromNumpy(x, w, name="h2", title="h2 title", nbins=100, xmin=-5, xmax=5)
        h1.Draw()

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


if __name__ == "__main__":
    unittest.main()
