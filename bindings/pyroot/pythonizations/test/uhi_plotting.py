"""
Tests to verify that th1 1D histograms conform to the UHI PlottableHistogram protocol.
"""

import numpy as np
import pytest
from conftest import _iterate_bins
from ROOT._pythonization._uhi.plotting import _shape
from uhi.typing.plottable import PlottableHistogram


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


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
