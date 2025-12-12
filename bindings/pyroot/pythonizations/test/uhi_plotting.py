"""
Tests to verify that th1 1D histograms conform to the UHI PlottableHistogram protocol.
"""

import numpy as np
import pytest
from ROOT._pythonization._uhi import _shape
from uhi.typing.plottable import PlottableHistogram


def _iterate_bins(hist, flow=False):
    ranges = [
        range(0 if flow else 1, hist.GetNbinsX() + (2 if flow else 1)),
        range(0 if flow else 1, hist.GetNbinsY() + (2 if flow else 1)) if hist.GetDimension() > 1 else [0],
        range(0 if flow else 1, hist.GetNbinsZ() + (2 if flow else 1)) if hist.GetDimension() > 2 else [0],
    ]
    yield from ((i, j, k)[: hist.GetDimension()] for i in ranges[0] for j in ranges[1] for k in ranges[2])


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
