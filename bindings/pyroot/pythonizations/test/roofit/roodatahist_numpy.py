import unittest

import ROOT

import numpy as np


class TestRooDataHistNumpy(unittest.TestCase):

    @staticmethod
    def _make_root_histo():
        # Create ROOT ROOT.TH1 filled with a Gaussian distribution
        hh = ROOT.TH1D("hh", "hh", 25, -10, 10)
        for i in range(100):
            hh.Fill(ROOT.gRandom.Gaus(0, 3), 0.5)
        return hh

    def test_to_numpy_and_from_numpy(self):
        """Test exporting to numpy and then importing back a RooDataHist."""

        hh = self._make_root_histo()

        # Declare observable x
        x = ROOT.RooRealVar("x", "x", -10, 10)

        datahist = ROOT.RooDataHist("data_hist", "data_hist", [x], Import=hh)

        hist, bin_edges = datahist.to_numpy()
        weights_squared_sum = datahist._weights_squared_sum()

        # We try both ways to pass bin edges: either via a full array, or with
        # the number of bins and the limits.
        datahist_1 = ROOT.RooDataHist.from_numpy(hist, [x], bins=bin_edges, weights_squared_sum=weights_squared_sum)
        datahist_2 = ROOT.RooDataHist.from_numpy(
            hist, [x], bins=[25], ranges=[(-10, 10)], weights_squared_sum=weights_squared_sum
        )

        def compare_to_ref(dh):
            hist_new, bin_edges_new = dh.to_numpy()
            np.testing.assert_allclose(hist_new, hist)
            np.testing.assert_allclose(bin_edges_new, bin_edges)
            np.testing.assert_allclose(dh._weights_squared_sum(), weights_squared_sum)

        compare_to_ref(datahist_1)
        compare_to_ref(datahist_2)


if __name__ == "__main__":
    unittest.main()
