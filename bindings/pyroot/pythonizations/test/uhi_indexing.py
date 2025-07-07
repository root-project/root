"""
Tests to verify that TH1 and derived histograms conform to the UHI Indexing interfaces (setting, accessing and slicing).
"""

import pytest
import ROOT
from ROOT._pythonization._uhi import _get_axis, _get_processed_slices, _overflow, _shape, _underflow
from ROOT.uhi import loc, overflow, rebin, sum, underflow


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


def _get_slice_indices(slices):
    import numpy as np

    ranges = [range(start, stop) for start, stop in slices]
    grids = np.meshgrid(*ranges, indexing="ij")
    return np.array(grids).reshape(len(slices), -1).T


class TestTH1Indexing:
    def test_access_with_bin_number(self, hist_setup):
        for index in [0, 8]:
            assert hist_setup[_get_index_for_dimension(hist_setup, index)] == hist_setup.GetBinContent(
                *_get_index_for_dimension(hist_setup, index + 1)
            )

    def test_access_with_data_coordinates(self, hist_setup):
        for value in [1, 4]:
            value_indices = _get_index_for_dimension(hist_setup, value)
            loc_indices = tuple(loc(idx) for idx in value_indices)
            assert hist_setup[loc_indices] == hist_setup.GetBinContent(hist_setup.FindBin(*value_indices))

    def test_access_flow_bins(self, hist_setup):
        for flow_type, index in [(underflow, 0), (overflow, 11)]:
            flow_indices = (flow_type,) * hist_setup.GetDimension()
            assert hist_setup[flow_indices] == hist_setup.GetBinContent(*_get_index_for_dimension(hist_setup, index))

    def test_access_with_len(self, hist_setup):
        len_indices = (len,) * hist_setup.GetDimension()
        bin_counts = (_get_axis(hist_setup, i).GetNbins() + 1 for i in range(hist_setup.GetDimension()))
        assert hist_setup[len_indices] == hist_setup.GetBinContent(*bin_counts)

    def test_access_with_ellipsis(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Access cannot be tested here")

        hist_copy = hist_setup[...]
        for bin_indices in _iterate_bins(hist_setup):
            assert hist_copy.GetBinContent(*bin_indices) == hist_setup.GetBinContent(*bin_indices)

    def test_setting_with_bin_number(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        for index in [0, 8]:
            hist_setup[_get_index_for_dimension(hist_setup, index)] = 60
            assert hist_setup.GetBinContent(*_get_index_for_dimension(hist_setup, index + 1)) == 60

    def test_setting_with_data_coordinates(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        value_indices = _get_index_for_dimension(hist_setup, 2)
        loc_indices = tuple(loc(idx) for idx in value_indices)
        hist_setup[loc_indices] = 70
        assert hist_setup.GetBinContent(hist_setup.FindBin(*value_indices)) == 70

    def test_setting_flow_bins(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        for flow_type, index in [(underflow, 0), (overflow, 11)]:
            flow_indices = (flow_type,) * hist_setup.GetDimension()
            hist_setup[flow_indices] = 85
            assert hist_setup.GetBinContent(*_get_index_for_dimension(hist_setup, index)) == 85

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

        hist_setup.Reset()
        hist_setup[...] = 3
        for bin_indices in _iterate_bins(hist_setup):
            assert hist_setup.GetBinContent(*bin_indices) == 3

        # Check that flow bins are not set
        for flow_type in [underflow, overflow]:
            flow_indices = (flow_type,) * hist_setup.GetDimension()
            assert hist_setup[flow_indices] == 0, (
                f"{hist_setup.values()}, {hist_setup[underflow]}, {hist_setup[overflow]}"
            )

    def _test_slices_match(self, hist_setup, slice_ranges, processed_slices):
        dim = hist_setup.GetDimension()
        slices, _ = _get_processed_slices(hist_setup, processed_slices[dim])
        expected_indices = _get_slice_indices(slices)
        sliced_hist = hist_setup[tuple(slice_ranges[dim])]

        for bin_indices in expected_indices:
            bin_indices = tuple(map(int, bin_indices))
            shifted_indices = []
            is_flow_bin = False
            for i, idx in enumerate(bin_indices):
                shift = slice_ranges[dim][i].start
                if callable(shift):
                    shift = shift(hist_setup, i)
                elif shift is None:
                    shift = 1
                else:
                    shift += 1

                shifted_idx = idx - shift + 1
                if shifted_idx <= 0 or shifted_idx == _overflow(hist_setup, i):
                    is_flow_bin = True
                    break

                shifted_indices.append(shifted_idx)

            if is_flow_bin:
                continue

            assert sliced_hist.GetBinContent(*tuple(shifted_indices)) == hist_setup.GetBinContent(*bin_indices)

    def test_slicing_with_endpoints(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        slice_ranges = {
            1: [slice(4, 7)],
            2: [slice(4, 7), slice(3, 6)],
            3: [slice(4, 7), slice(3, 6), slice(2, 5)],
        }
        processed_slices = {
            1: [slice(5, 8)],
            2: [slice(5, 8), slice(4, 7)],
            3: [slice(5, 8), slice(4, 7), slice(3, 6)],
        }
        self._test_slices_match(hist_setup, slice_ranges, processed_slices)

    def test_slicing_without_endpoints(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        processed_slices = {
            1: [slice(0, 8)],
            2: [slice(0, 8), slice(0, 8)],
            3: [slice(0, 8), slice(0, 8), slice(3, 6)],
        }
        slice_ranges = {
            1: [slice(None, 7)],
            2: [slice(None, 7), slice(None, 7)],
            3: [slice(None, 7), slice(None, 7), slice(2, 5)],
        }
        self._test_slices_match(hist_setup, slice_ranges, processed_slices)

    def test_slicing_with_data_coordinates(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        processed_slices = {
            1: [slice(hist_setup.FindBin(2), 11)],
            2: [slice(hist_setup.FindBin(2) - 1, 11), slice(2, 11)],
            3: [
                slice(hist_setup.FindBin(2), 11),
                slice(2, 11),
                slice(2, 11),
            ],
        }
        slice_ranges = {
            1: [slice(loc(2), None)],
            2: [slice(loc(2), None), slice(3, None)],
            3: [slice(loc(2), None), slice(3, None), slice(3, None)],
        }
        self._test_slices_match(hist_setup, slice_ranges, processed_slices)

    def test_slicing_over_everything_with_action_rebin(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        rebin_indices = tuple(slice(None, None, rebin(2)) for _ in range(hist_setup.GetDimension()))
        sliced_hist = hist_setup[rebin_indices]
        assert sliced_hist.GetNbinsX() == hist_setup.GetNbinsX() // 2
        assert sliced_hist.GetDimension() < 2 or sliced_hist.GetNbinsY() == hist_setup.GetNbinsY() // 2
        assert sliced_hist.GetDimension() < 3 or sliced_hist.GetNbinsZ() == hist_setup.GetNbinsZ() // 2

    def test_slicing_over_everything_with_action_sum(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        dim = hist_setup.GetDimension()

        if dim == 1:
            full_integral = hist_setup[::sum]
            assert full_integral == hist_setup.Integral(_underflow(hist_setup, 0), _overflow(hist_setup, 0))

            integral = hist_setup[0:len:sum]
            assert integral == hist_setup.Integral()

        if dim == 2:
            sum_x = hist_setup[:, ::sum]
            assert isinstance(sum_x, ROOT.TH1)
            assert sum_x.GetNbinsX() == hist_setup.GetNbinsX()

            sum_y = hist_setup[::sum, :]
            assert isinstance(sum_y, ROOT.TH1)
            assert sum_y.GetNbinsX() == hist_setup.GetNbinsY()

        if dim == 3:
            sum_z = hist_setup[:, :, ::sum]
            assert isinstance(sum_z, ROOT.TH2)

            sum_yz = hist_setup[:, ::sum, ::sum]
            assert isinstance(sum_yz, ROOT.TH1)

            sum_xz = hist_setup[::sum, :, ::sum]
            assert isinstance(sum_xz, ROOT.TH1)

            sum_xy = hist_setup[::sum, ::sum, :]
            assert isinstance(sum_xy, ROOT.TH1)

    def test_slicing_with_action_rebin_and_sum(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Slicing cannot be tested here")

        dim = hist_setup.GetDimension()

        if dim == 1:
            sliced_hist_rebin = hist_setup[5 : 9 : rebin(2)]
            assert isinstance(sliced_hist_rebin, ROOT.TH1)
            assert sliced_hist_rebin.GetNbinsX() == 2

            sliced_hist_sum = hist_setup[5:9:sum]
            assert isinstance(sliced_hist_sum, float)
            assert sliced_hist_sum == hist_setup.Integral(6, 9)

        if dim == 2:
            sliced_hist = hist_setup[:: rebin(2), 5:9:sum]
            assert isinstance(sliced_hist, ROOT.TH1)
            assert sliced_hist.GetNbinsX() == hist_setup.GetNbinsX() // 2

        if dim == 3:
            sliced_hist = hist_setup[:: rebin(2), ::sum, 3 : 9 : rebin(3)]
            assert isinstance(sliced_hist, ROOT.TH2)
            assert sliced_hist.GetNbinsX() == hist_setup.GetNbinsX() // 2
            assert sliced_hist.GetNbinsY() == 2

    def test_slicing_with_dict_syntax(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        slices = {dim: slice(4, 7) for dim in range(hist_setup.GetDimension())}
        sliced_hist = hist_setup[tuple(slices.values())]
        dict_sliced_hist = hist_setup[slices]
        for bin_indices in _iterate_bins(hist_setup):
            assert sliced_hist.GetBinContent(*bin_indices) == dict_sliced_hist.GetBinContent(*bin_indices)

    def test_integral_full_slice(self, hist_setup):
        if _special_setting(hist_setup):
            pytest.skip("Setting cannot be tested here")

        # Check if slicing over everything preserves the integral
        sliced_hist = hist_setup[...]

        assert hist_setup.Integral() == pytest.approx(sliced_hist.Integral(), rel=10e-6)

    def test_statistics_slice(self, hist_setup):
        if _special_setting(hist_setup) or isinstance(hist_setup, (ROOT.TH1C, ROOT.TH2C, ROOT.TH3C)):
            pytest.skip("Setting cannot be tested here")

        # Check if slicing over everything preserves the statistics
        sliced_hist_full = hist_setup[...]

        assert hist_setup.GetEffectiveEntries() == pytest.approx(sliced_hist_full.GetEffectiveEntries())
        assert sliced_hist_full.GetEntries() == pytest.approx(sliced_hist_full.GetEffectiveEntries())
        assert hist_setup.Integral() == pytest.approx(sliced_hist_full.Integral())

        # Check if slicing over a range updates the statistics
        dim = hist_setup.GetDimension()
        [_get_axis(hist_setup, i).SetRange(3, 7) for i in range(dim)]
        slice_indices = tuple(slice(2, 7) for _ in range(dim))
        sliced_hist = hist_setup[slice_indices]

        assert hist_setup.Integral() == sliced_hist.Integral()
        assert hist_setup.GetEffectiveEntries() == pytest.approx(sliced_hist.GetEffectiveEntries())
        assert sliced_hist.GetEntries() == pytest.approx(sliced_hist.GetEffectiveEntries())
        assert hist_setup.GetStdDev() == pytest.approx(sliced_hist.GetStdDev(), rel=10e-5)
        assert hist_setup.GetMean() == pytest.approx(sliced_hist.GetMean(), rel=10e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
