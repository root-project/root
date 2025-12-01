import unittest

import ROOT


class FillWithArrayLike(unittest.TestCase):
    """
    Test for the FillN method of TH1 and subclasses, which fills
    the histogram with array-like input.
    """

    def _run_fill_test(self, data):
        import numpy as np

        # Convert once for the reference FillN call
        data_np = np.asanyarray(data, dtype=np.float64)
        n = len(data_np)

        # Create histograms
        nbins = 5
        verbose_hist = ROOT.TH1F("verbose_hist", "verbose_hist", nbins, 0, 10)
        simple_hist = ROOT.TH1F("simple_hist", "simple_hist", nbins, 0, 10)

        # Test filling without weights
        verbose_hist.FillN(n, data_np, np.ones(n))
        simple_hist.Fill(data)

        for i in range(nbins):
            self.assertAlmostEqual(verbose_hist.GetBinContent(i), simple_hist.GetBinContent(i))

        # Test filling with weights
        weights_np = np.linspace(0.1, 0.9, n)
        weights = list(weights_np)  # also array-like

        verbose_hist.FillN(n, data_np, weights_np)
        simple_hist.Fill(data, weights)

        for i in range(nbins):
            self.assertAlmostEqual(verbose_hist.GetBinContent(i), simple_hist.GetBinContent(i))

        # Test mismatched weight size
        with self.assertRaises(ValueError):
            simple_hist.Fill(data, [0.1, 0.2, 0.3])  # too short

    # Run with different inputs
    def test_fill_arraylike(self):
        import numpy as np

        inputs = [
            np.array([1.0, 2, 2, 3, 3, 3, 4, 4, 5]),  # numpy
            [1.0, 2, 2, 3, 3, 3, 4, 4, 5],  # list
            range(9),  # range
        ]

        for input_data in inputs:
            with self.subTest(input_type=type(input_data).__name__):
                self._run_fill_test(input_data)


if __name__ == "__main__":
    unittest.main()
