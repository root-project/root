import unittest

import ROOT


class FillWithNumpyArray(unittest.TestCase):
    """
    Test for the FillN method of TH1 and subclasses, which fills
    the histogram with a numpy array.
    """

    # Tests
    def test_fill(self):
        import numpy as np
        # Create sample data
        data = np.array([1., 2, 2, 3, 3, 3, 4, 4, 5])
        # Create histograms
        nbins = 5
        min_val = 0
        max_val = 10
        verbose_hist = ROOT.TH1F("verbose_hist", "verbose_hist", nbins, min_val, max_val)
        simple_hist = ROOT.TH1F("simple_hist", "simple_hist", nbins, min_val, max_val)
        # Fill histograms
        verbose_hist.FillN(len(data), data, np.ones(len(data)))
        simple_hist.Fill(data)
        # Test if the histograms have the same content
        for i in range(nbins):
            self.assertAlmostEqual(verbose_hist.GetBinContent(i), simple_hist.GetBinContent(i))
        # Test filling with weights
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        verbose_hist.FillN(len(data), data, weights)
        simple_hist.Fill(data, weights)
        for i in range(nbins):
            self.assertAlmostEqual(verbose_hist.GetBinContent(i), simple_hist.GetBinContent(i))
        # Test filling with weights with a different length
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            simple_hist.Fill(data, weights)

if __name__ == '__main__':
    unittest.main()
