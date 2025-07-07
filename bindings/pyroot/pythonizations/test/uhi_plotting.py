"""
Tests to verify that th1 1D histograms conform to the UHI PlottableHistogram protocol.
"""

import numpy as np
import pytest
from ROOT._pythonization._uhi import _shape
from uhi.typing.plottable import PlottableHistogram


def _iterate_bins(hist):
    dim = hist.GetDimension()
    for i in range(1, hist.GetNbinsX() + 1):
        for j in range(1, hist.GetNbinsY() + 1) if dim > 1 else [None]:
            for k in range(1, hist.GetNbinsZ() + 1) if dim > 2 else [None]:
                yield tuple(filter(None, (i, j, k)))


class TestTH1Plotting:
    def test_isinstance_plottablehistogram(self, hist_setup):
        assert isinstance(hist_setup, PlottableHistogram)

    def test_histogram_values(self, hist_setup):
        values = np.array(
            [hist_setup.GetBinContent(*bin_indices) for bin_indices in _iterate_bins(hist_setup)]
        ).reshape(_shape(hist_setup, include_flow_bins=False))
        assert np.array_equal(hist_setup.values(), values)


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
