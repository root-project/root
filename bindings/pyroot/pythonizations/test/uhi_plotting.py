"""
Tests to verify that th1 1D histograms conform to the UHI PlottableHistogram protocol.
"""

import numpy as np
import pytest
import ROOT
from conftest import _iterate_bins
from ROOT._pythonization._uhi.plotting import _shape
from uhi.typing.plottable import PlottableHistogram


def _not_writable(hist):
    # For these classes, the values() method always returns a copy so writable=True is not supported.
    return isinstance(hist, (ROOT.TH1C, ROOT.TH2C, ROOT.TH3C, ROOT.TProfile, ROOT.TProfile2D, ROOT.TProfile3D))


class TestTH1Plotting:
    def test_isinstance_plottablehistogram(self, hist_setup):
        assert isinstance(hist_setup, PlottableHistogram)

    def test_histogram_values(self, hist_setup):
        values = np.array(
            [hist_setup.GetBinContent(*bin_indices) for bin_indices in _iterate_bins(hist_setup)]
        ).reshape(_shape(hist_setup, flow=False))
        assert np.array_equal(hist_setup.values(), values)

    def test_histogram_values_flow(self, hist_setup):
        values_flow = np.array(
            [hist_setup.GetBinContent(*bin_indices) for bin_indices in _iterate_bins(hist_setup, flow=True)]
        ).reshape(_shape(hist_setup, flow=True))
        assert np.array_equal(hist_setup.values(flow=True), values_flow)

    def test_histogram_values_writable(self, hist_setup):
        # default: read-only
        values = hist_setup.values()
        assert values.flags.writeable is False

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            values[0] = 42

        # opt-in: writable
        if _not_writable(hist_setup):
            with pytest.raises(TypeError, match="values\\(writable=True\\) is not supported for this histogram type"):
                hist_setup.values(writable=True)
        else:
            values_w = hist_setup.values(writable=True)
            assert values_w.flags.writeable is True


class TestProfilePlotting:
    @staticmethod
    def _make_profile(name, weighted=False):
        profile = ROOT.TProfile(name, "", 4, 0, 4)
        if weighted:
            profile.Sumw2()
        for x, y, w in [(0.5, 1, 2), (0.5, 3, 1), (1.5, 2, 1), (1.5, 2, 1), (1.5, 5, 1)]:
            profile.Fill(x, y, w if weighted else 1)
        return profile

    def test_counts(self):
        assert np.array_equal(self._make_profile("p_counts").counts(), [2, 3, 0, 0])

        # Effective number of entries (sum of weights)^2 / (sum of weights^2)
        weighted_counts = self._make_profile("p_counts_weighted", weighted=True).counts()
        assert weighted_counts == pytest.approx([9 / 5, 3, 0, 0])

    def test_variances(self):
        # Variance of the mean of y in each bin, NaN for the bins with less than two entries
        variances = self._make_profile("p_variances").variances()
        assert np.allclose(variances, [1, 1, np.nan, np.nan], equal_nan=True)

        weighted_variances = self._make_profile("p_variances_weighted", weighted=True).variances()
        assert np.allclose(weighted_variances, [2 / 3, 1, np.nan, np.nan], equal_nan=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
