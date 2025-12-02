import unittest

import ROOT


class FillWithArrayLike(unittest.TestCase):
    """
    Test for the FillN method of TH2, which fills
    the histogram with array-like input.
    """

    def _run_fill_test(self, x_data, y_data):
        import numpy as np

        # Convert once for the reference FillN call
        x_np = np.asanyarray(x_data, dtype=np.float64)
        y_np = np.asanyarray(y_data, dtype=np.float64)
        n = len(x_np)

        # Create histograms
        nbins = 5
        verbose_hist = ROOT.TH2F("verbose_hist", "verbose_hist", nbins, 0, 10, nbins, 0, 10)
        simple_hist = ROOT.TH2F("simple_hist", "simple_hist", nbins, 0, 10, nbins, 0, 10)

        # Test filling without weights
        verbose_hist.FillN(n, x_np, y_np, np.ones(n))
        simple_hist.Fill(x_data, y_data)

        for i in range(nbins):
            for j in range(nbins):
                self.assertAlmostEqual(verbose_hist.GetBinContent(i, j), simple_hist.GetBinContent(i, j))

        # Test filling with weights
        weights_np = np.linspace(0.1, 0.9, n)
        weights = list(weights_np)

        verbose_hist.FillN(n, x_np, y_np, weights_np)
        simple_hist.Fill(x_data, y_data, weights)

        for i in range(nbins):
            for j in range(nbins):
                self.assertAlmostEqual(verbose_hist.GetBinContent(i, j), simple_hist.GetBinContent(i, j))

        # Test mismatched weight size
        with self.assertRaises(ValueError):
            simple_hist.Fill(x_data, y_data, [0.1, 0.2, 0.3])  # too short

    # Run with different inputs
    def test_fill_arraylike(self):
        import numpy as np

        inputs = [
            (np.array([1.0, 2, 3, 4, 5]), np.array([5.0, 4, 3, 2, 1])),
            ([1.0, 2, 3, 4, 5], [5.0, 4, 3, 2, 1]),
            (range(1, 6), range(5, 0, -1)),
            ([1.0, 2, 3, 4, 5], np.array([5.0, 4, 3, 2, 1])),
        ]

        for x_data, y_data in inputs:
            with self.subTest(input_type=(type(x_data).__name__, type(y_data).__name__)):
                self._run_fill_test(x_data, y_data)


if __name__ == "__main__":
    unittest.main()
