"""
Tests to verify that TH1 histograms conform to the UHI setting and accessing interfaces.
"""

import pytest
import ROOT
from ROOT._pythonization._th1 import _th1_derived_classes_to_pythonize
from ROOT._pythonization._th2 import _th2_derived_classes_to_pythonize
from ROOT._pythonization._th3 import _th3_derived_classes_to_pythonize
from ROOT._pythonization._uhi import _shape

th1_classes = {
    ("test_hist", "Test Histogram", 10, 1, 4): {
        "classes": [getattr(ROOT, klass) for klass in _th1_derived_classes_to_pythonize],
        "fill": lambda hist: hist.Fill(ROOT.gRandom.Uniform(0, 5), 1.0),
    },
    ("test_hist", "Test Histogram", 10, 1, 4, 10, 1, 4): {
        "classes": [getattr(ROOT, klass) for klass in _th2_derived_classes_to_pythonize],
        "fill": lambda hist: hist.Fill(ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5), 1.0),
    },
    ("test_hist", "Test Histogram", 10, 1, 4, 10, 1, 4, 10, 1, 4): {
        "classes": [getattr(ROOT, klass) for klass in _th3_derived_classes_to_pythonize],
        "fill": lambda hist: hist.Fill(
            ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5), 1.0
        ),
    },
}


@pytest.fixture(
    params=[
        (constructor_args, hist_class, hist_config["fill"])
        for constructor_args, hist_config in th1_classes.items()
        for hist_class in hist_config["classes"]
    ],
    scope="class",
    ids=lambda param: param[1].__name__,
)
def hist_setup(request):
    constructor_args, hist_class, fill = request.param
    hist = hist_class(*constructor_args)
    for _ in range(10000):
        fill(hist)
    yield hist


def special_setting(hist):
    # For these classes, SetBinContent works differently than for other classes,
    # as in it does not set the bin content to the specified value, but does some other calculations
    # for that, these classes will be tested differently
    return isinstance(hist, (ROOT.TProfile, ROOT.TProfile2D, ROOT.TProfile2Poly, ROOT.TProfile3D, ROOT.TH1K))


def get_index_for_dimension(hist, index):
    return (index,) * hist.GetDimension()


class TestTH1Indexing:
    def test_access_with_bin_number(self, hist_setup):
        first_index = get_index_for_dimension(hist_setup, 1)
        assert hist_setup[first_index] == hist_setup.GetBinContent(*first_index)
        last_index = get_index_for_dimension(hist_setup, hist_setup.GetNcells() - 2)
        assert hist_setup[last_index] == hist_setup.GetBinContent(*last_index)

    def test_access_with_data_coordinates(self, hist_setup):
        value_index = get_index_for_dimension(hist_setup, 2)
        assert hist_setup[ROOT.loc(*value_index)] == hist_setup.GetBinContent(hist_setup.FindBin(*value_index))

    def test_access_flow_bins(self, hist_setup):
        assert hist_setup[ROOT.underflow] == hist_setup.GetBinContent(0)
        assert hist_setup[ROOT.overflow] == hist_setup.GetBinContent(hist_setup.GetNcells() - 1)

    def test_access_with_ellipsis(self, hist_setup):
        import numpy as np

        expected = np.array([hist_setup.GetBinContent(i) for i in range(np.prod(_shape(hist_setup)))])
        assert all(hist_setup[...].flatten() == expected)

    def test_setting_with_bin_number(self, hist_setup):
        if special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")
        first_index = (1,) if isinstance(hist_setup, ROOT.TProfile2Poly) else get_index_for_dimension(hist_setup, 1)
        hist_setup[first_index] = 60
        assert hist_setup.GetBinContent(*first_index) == 60

    def test_setting_with_data_coordinates(self, hist_setup):
        if special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")
        value_index = get_index_for_dimension(hist_setup, 2)
        hist_setup[ROOT.loc(*value_index)] = 10
        assert hist_setup.GetBinContent(hist_setup.FindBin(*value_index)) == 10

    def test_setting_flow_bins(self, hist_setup):
        if special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")
        hist_setup[ROOT.underflow] = 10
        hist_setup[ROOT.overflow] = 10
        assert hist_setup.GetBinContent(0) == 10
        assert hist_setup.GetBinContent(hist_setup.GetNcells() - 1) == 10

    def test_setting_with_array(self, hist_setup):
        import numpy as np

        if special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")
        expected = np.array([3] * np.prod(_shape(hist_setup)))
        expected = expected.reshape(_shape(hist_setup))
        hist_setup[...] = expected
        assert np.allclose(hist_setup[...], expected), hist_setup[...]


if __name__ == "__main__":
    pytest.main(args=[__file__])
