"""
Tests to verify that th1 1D histograms conform to the UHI PlottableHistogram protocol.
"""

import ROOT
from ROOT._pythonization._th1 import _th1_derived_classes_to_pythonize

import pytest
from uhi.typing.plottable import PlottableHistogram
import numpy as np

th1_classes = [getattr(ROOT, klass) for klass in _th1_derived_classes_to_pythonize]

@pytest.mark.parametrize("hist_type", th1_classes)
class TestTH1Plotting:
    @pytest.fixture
    def hist_setup(self, hist_type):
        hist = hist_type("h", "h", 10, 0, 5)
        for _ in range(100):
            hist.Fill(ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5))
        yield hist

    def test_isinstance_plottablehistogram(self, hist_setup):
        assert isinstance(hist_setup, PlottableHistogram)

    def test_histogram_values(self, hist_setup):
        assert np.allclose(hist_setup.values(), [hist_setup.GetBinContent(i) for i in range(1, hist_setup.GetNbinsX() + 1)])

if __name__ == "__main__":
    pytest.main(args=[__file__])