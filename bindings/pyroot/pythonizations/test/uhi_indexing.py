"""
Tests to verify that TH1 histograms conform to the UHI setting and accessing interfaces.
"""

import pytest
import ROOT
from ROOT._pythonization._th1 import _th1_derived_classes_to_pythonize
from ROOT._pythonization._th2 import _th2_derived_classes_to_pythonize
from ROOT._pythonization._th3 import _th3_derived_classes_to_pythonize
from ROOT._pythonization._uhi import _get_processed_slices, _get_slice_indices, _shape

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


def _special_setting(hist):
    # For these classes, SetBinContent works differently than for other classes,
    # as in it does not set the bin content to the specified value, but does some other calculations
    # for that, these classes will be tested differently
    return isinstance(hist, (ROOT.TProfile, ROOT.TProfile2D, ROOT.TProfile2Poly, ROOT.TProfile3D, ROOT.TH1K))


def _get_index_for_dimension(hist, index):
    return (index,) * hist.GetDimension()


def _iterate_bins(hist):
    dim = hist.GetDimension()
    for i in range(1, hist.GetNbinsX() + 1):
        for j in range(1, hist.GetNbinsY() + 1) if dim > 1 else [None]:
            for k in range(1, hist.GetNbinsZ() + 1) if dim > 2 else [None]:
                yield tuple(filter(None, (i, j, k)))


class TestTH1Indexing:
    def test_access_with_bin_number(self, hist_setup):
        for index in [1, 9]:
            bin_index = _get_index_for_dimension(hist_setup, index)
            assert hist_setup[bin_index] == hist_setup.GetBinContent(*bin_index)

    def test_access_with_data_coordinates(self, hist_setup):
        for value in [1, 4]:
            value_indices = _get_index_for_dimension(hist_setup, value)
            loc_indices = tuple(ROOT.loc(idx) for idx in value_indices)
            assert hist_setup[loc_indices] == hist_setup.GetBinContent(hist_setup.FindBin(*value_indices))

    def test_access_flow_bins(self, hist_setup):
        for flow_type, index in [(ROOT.underflow, 0), (ROOT.overflow, 11)]:
            flow_indices = [flow_type] * hist_setup.GetDimension()
            assert hist_setup[tuple(flow_indices)] == hist_setup.GetBinContent(
                *_get_index_for_dimension(hist_setup, index)
            )

    def test_access_with_ellipsis(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Access cannot be tested here")

        hist_copy = hist_setup[...]
        for bin_indices in _iterate_bins(hist_setup):
            assert hist_copy.GetBinContent(*bin_indices) == hist_setup.GetBinContent(*bin_indices)

    def test_setting_with_bin_number(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        first_index = (1,) if isinstance(hist_setup, ROOT.TProfile2Poly) else _get_index_for_dimension(hist_setup, 1)
        hist_setup[first_index] = 60
        assert hist_setup.GetBinContent(*first_index) == 60

    def test_setting_with_data_coordinates(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        value_indices = _get_index_for_dimension(hist_setup, 2)
        loc_indices = tuple(ROOT.loc(idx) for idx in value_indices)
        hist_setup[loc_indices] = 70
        assert hist_setup.GetBinContent(hist_setup.FindBin(*value_indices)) == 70

    def test_setting_flow_bins(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        for flow_type, index in [(ROOT.underflow, 0), (ROOT.overflow, 11)]:
            flow_indices = [flow_type] * hist_setup.GetDimension()
            hist_setup[tuple(flow_indices)] = 85
            assert hist_setup.GetBinContent(*_get_index_for_dimension(hist_setup, index)) == 85

    def _test_slices_match(self, hist_setup, slice_ranges, processed_slices):
        dim = hist_setup.GetDimension()
        slices, _ = _get_processed_slices(hist_setup, processed_slices[dim])
        expected_indices = _get_slice_indices(slices)
        sliced_hist = hist_setup[tuple(slice_ranges[dim])]

        for bin_indices in expected_indices:
            bin_indices = tuple(map(int, bin_indices))
            assert sliced_hist.GetBinContent(*bin_indices) == hist_setup.GetBinContent(*bin_indices)

        for bin_indices in _iterate_bins(hist_setup):
            if list(bin_indices) not in expected_indices.tolist():
                assert sliced_hist.GetBinContent(*bin_indices) == 0

    def test_setting_with_array(self, hist_setup):
        import numpy as np

        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        expected = np.full(_shape(hist_setup), 3, dtype=np.int64)
        hist_setup[...] = expected
        for bin_indices in _iterate_bins(hist_setup):
            assert hist_setup.GetBinContent(*bin_indices) == 3

    def test_setting_with_scalar(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        hist_setup[...] = 3
        for bin_indices in _iterate_bins(hist_setup):
            assert hist_setup.GetBinContent(*bin_indices) == 3

    def test_slicing_with_endpoints(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        slice_ranges = {
            1: [slice(4, 7)],
            2: [slice(4, 7), slice(3, 6)],
            3: [slice(4, 7), slice(3, 6), slice(2, 5)],
        }
        self._test_slices_match(hist_setup, slice_ranges, slice_ranges)

    def test_slicing_without_endpoints(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        processed_slices = {
            1: [slice(0, 7)],
            2: [slice(0, 7), slice(3, 11)],
            3: [slice(0, 7), slice(3, 11), slice(2, 5)],
        }
        slice_ranges = {
            1: [slice(None, 7)],
            2: [slice(None, 7), slice(3, None)],
            3: [slice(None, 7), slice(3, None), slice(2, 5)],
        }
        self._test_slices_match(hist_setup, slice_ranges, processed_slices)

    def test_slicing_with_data_coordinates(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        processed_slices = {
            1: [slice(hist_setup.FindBin(2), 11)],
            2: [slice(hist_setup.FindBin(2), 11), slice(hist_setup.FindBin(3), 11)],
            3: [
                slice(hist_setup.FindBin(2), 11),
                slice(hist_setup.FindBin(3), 11),
                slice(hist_setup.FindBin(1.5), 11),
            ],
        }
        slice_ranges = {
            1: [slice(ROOT.loc(2), None)],
            2: [slice(ROOT.loc(2), None), slice(ROOT.loc(3), None)],
            3: [slice(ROOT.loc(2), None), slice(ROOT.loc(3), None), slice(ROOT.loc(1.5), None)],
        }
        self._test_slices_match(hist_setup, slice_ranges, processed_slices)

    def test_slicing_with_action_rebin(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        rebin_indices = tuple(slice(None, None, ROOT.rebin(2)) for _ in range(hist_setup.GetDimension()))
        sliced_hist = hist_setup[rebin_indices]
        assert sliced_hist.GetNbinsX() == hist_setup.GetNbinsX() // 2
        assert sliced_hist.GetDimension() < 2 or sliced_hist.GetNbinsY() == hist_setup.GetNbinsY() // 2
        assert sliced_hist.GetDimension() < 3 or sliced_hist.GetNbinsZ() == hist_setup.GetNbinsZ() // 2

    def test_slicing_with_action_sum(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        dim = hist_setup.GetDimension()

        if dim == 1:
            integral = hist_setup[:: ROOT.sum]
            assert integral == hist_setup.Integral()

        if dim == 2:
            sum_x = hist_setup[:, :: ROOT.sum]
            assert isinstance(sum_x, ROOT.TH1)
            assert sum_x.GetNbinsX() == hist_setup.GetNbinsX()

            sum_y = hist_setup[:: ROOT.sum, :]
            assert isinstance(sum_y, ROOT.TH1)
            assert sum_y.GetNbinsX() == hist_setup.GetNbinsY()

        if dim == 3:
            sum_z = hist_setup[:, :, :: ROOT.sum]
            assert isinstance(sum_z, ROOT.TH1)

            sum_yz = hist_setup[:, :: ROOT.sum, :: ROOT.sum]
            assert isinstance(sum_yz, ROOT.TH2)

            sum_xz = hist_setup[:: ROOT.sum, :, :: ROOT.sum]
            assert isinstance(sum_xz, ROOT.TH2)

            sum_xy = hist_setup[:: ROOT.sum, :: ROOT.sum, :]
            assert isinstance(sum_xy, ROOT.TH2)

    def test_slicing_with_dict_syntax(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        slices = {dim: slice(4, 7) for dim in range(hist_setup.GetDimension())}
        sliced_hist = hist_setup[tuple(slices.values())]
        dict_sliced_hist = hist_setup[slices]
        for bin_indices in _iterate_bins(hist_setup):
            assert sliced_hist.GetBinContent(*bin_indices) == dict_sliced_hist.GetBinContent(*bin_indices)


if __name__ == "__main__":
    pytest.main(args=[__file__])
