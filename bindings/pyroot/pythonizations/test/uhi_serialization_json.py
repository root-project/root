"""
Tests to verify that TH1 and derived histograms conform to the UHI Serialization interfaces.
"""

"""
Tests to verify that ROOT histograms can be serialized to and from
the UHI JSON IR (round-trip) for 1D, 2D, and 3D histograms.
"""

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import ROOT  # noqa: E402
import uhi.io.json  # noqa: E402
from conftest import _iterate_bins  # noqa: E402


def _bin_contents(hist, flow=False):
    return np.array([hist.GetBinContent(*idx) for idx in _iterate_bins(hist, flow=flow)])


def _roundtrip(hist):
    """
    ROOT histogram -> JSON -> UHI IR -> ROOT histogram
    """
    json_str = json.dumps(hist, default=uhi.io.json.default)
    ir = json.loads(json_str, object_hook=uhi.io.json.object_hook)
    return hist.__class__(ir)


# Tests
class TestTH1Serialization:
    def test_roundtrip_1d(self):
        h = ROOT.TH1D("h1", "h1", 10, -5, 5)
        h[...] = np.arange(10)

        # test constructor based serialization roundtrip
        h_loaded = _roundtrip(h)

        assert np.array_equal(
            _bin_contents(h, flow=False),
            _bin_contents(h_loaded, flow=False),
        )

        # test classmethod based serialization roundtrip
        ir = h._to_uhi_()
        h_loaded_cls = ROOT.TH1D._from_uhi_(ir)
        assert np.array_equal(
            _bin_contents(h, flow=False),
            _bin_contents(h_loaded_cls, flow=False),
        )

        # test that name is preserved
        assert h.GetName() == h_loaded.GetName() == h_loaded_cls.GetName()

    def test_roundtrip_1d_flow(self):
        h = ROOT.TH1D("h1f", "h1f", 5, 0, 5)
        h.Fill(-1)
        h.Fill(6)
        h[...] = np.arange(5)

        # test constructor based serialization roundtrip
        h_loaded = _roundtrip(h)

        assert np.array_equal(
            _bin_contents(h, flow=True),
            _bin_contents(h_loaded, flow=True),
        )

        # test classmethod based serialization roundtrip
        ir = h._to_uhi_()
        h_loaded_cls = ROOT.TH1D._from_uhi_(ir)
        assert np.array_equal(
            _bin_contents(h, flow=True),
            _bin_contents(h_loaded_cls, flow=True),
        )

        # test that name is preserved
        assert h.GetName() == h_loaded.GetName() == h_loaded_cls.GetName()

    def test_axis_does_not_depend_on_range(self):
        h = ROOT.TH1D("h1_range", "h1_range", 10, 0, 10)
        h.GetXaxis().SetRange(3, 5)

        axis = h._to_uhi_()["axes"][0]

        assert (axis["lower"], axis["upper"], axis["bins"]) == (0, 10, 10)

    def test_variable_axis(self):
        edges = [0.0, 1.0, 5.0, 10.0]
        h = ROOT.TH1D("h1_variable", "h1_variable", 3, np.array(edges))

        axis = h._to_uhi_()["axes"][0]

        assert axis["type"] == "variable"
        assert axis["edges"] == edges

    def test_profile_storage(self):
        profile = ROOT.TProfile("p_storage", "p_storage", 4, 0, 4)
        for x, y in [(0.5, 1), (0.5, 3), (1.5, 2), (1.5, 2), (1.5, 5)]:
            profile.Fill(x, y)

        storage = profile._to_uhi_()["storage"]

        # The variances of the storage are the unbiased variances of y
        assert storage["type"] == "mean"
        assert np.array_equal(storage["counts"], [0, 2, 3, 0, 0, 0])
        assert np.allclose(storage["variances"], [0, 2, 3, 0, 0, 0])

    def test_weighted_profile_storage(self):
        profile = ROOT.TProfile("p_storage_weighted", "p_storage_weighted", 4, 0, 4)
        profile.Sumw2()
        for x, y, w in [(0.5, 1, 2), (0.5, 3, 1)]:
            profile.Fill(x, y, w)

        storage = profile._to_uhi_()["storage"]

        assert storage["type"] == "weighted_mean"
        assert np.array_equal(storage["sum_of_weights"], [0, 3, 0, 0, 0, 0])
        assert np.array_equal(storage["sum_of_weights_squared"], [0, 5, 0, 0, 0, 0])
        assert np.allclose(storage["variances"], [0, 2, 0, 0, 0, 0])

    def test_uhi_protocol_object(self):
        values = np.arange(12.0)

        class Dummy:
            def _to_uhi_(self):
                return {
                    "uhi_schema": 1,
                    "axes": [
                        {
                            "type": "regular",
                            "lower": -5.0,
                            "upper": 5.0,
                            "bins": 10,
                            "underflow": True,
                            "overflow": True,
                            "circular": False,
                        }
                    ],
                    "storage": {"type": "double", "values": values},
                }

        h = ROOT.TH1D(Dummy())

        assert np.array_equal(_bin_contents(h, flow=True), values)

    def test_invalid_schema(self):
        h = ROOT.TH1D("h_invalid", "h_invalid", 10, -5, 5)
        ir = h._to_uhi_()
        ir["uhi_schema"] = 999  # only version 1 currently supported

        with pytest.raises(ValueError, match="Unsupported UHI schema version: 999"):
            ROOT.TH1D(ir)

        with pytest.raises(ValueError, match="Unsupported UHI schema version: 999"):
            ROOT.TH1D._from_uhi_(ir)

        with pytest.raises(ValueError, match="missing required keys 'axes' and 'storage'"):
            ir_invalid = {"uhi_schema": 1}
            ROOT.TH1D(ir_invalid)


class TestTH2Serialization:
    def test_roundtrip_2d(self):
        h = ROOT.TH2D("h2", "h2", 4, 0, 4, 3, 0, 3)

        values = np.arange(12).reshape(4, 3)
        h[...] = values

        h_loaded = _roundtrip(h)

        assert np.array_equal(
            _bin_contents(h, flow=False),
            _bin_contents(h_loaded, flow=False),
        )

        assert h.GetName() == h_loaded.GetName()

    def test_roundtrip_2d_flow(self):
        h = ROOT.TH2D("h2f", "h2f", 3, 0, 3, 2, 0, 2)
        h.Fill(-1, 1)
        h.Fill(4, 3)
        h[...] = np.arange(6).reshape(3, 2)

        h_loaded = _roundtrip(h)

        assert np.array_equal(
            _bin_contents(h, flow=True),
            _bin_contents(h_loaded, flow=True),
        )

        assert h.GetName() == h_loaded.GetName()

    def test_weighted_storage_includes_flow(self):
        h = ROOT.TH2D("h2_weighted", "h2_weighted", 3, 0, 3, 2, 0, 2)
        h.Sumw2()
        h.Fill(-1, 1, 2.0)
        h.Fill(1, 1, 3.0)

        storage = h._to_uhi_()["storage"]

        assert storage["type"] == "weighted"
        assert np.array_equal(storage["values"], h.values(flow=True))
        assert np.array_equal(storage["variances"], h.variances(flow=True))


class TestTH3Serialization:
    def test_roundtrip_3d(self):
        h = ROOT.TH3D("h3", "h3", 3, 0, 3, 2, 0, 2, 2, 0, 2)

        values = np.arange(12).reshape(3, 2, 2)
        h[...] = values

        h_loaded = _roundtrip(h)

        assert np.array_equal(
            _bin_contents(h, flow=False),
            _bin_contents(h_loaded, flow=False),
        )

        assert h.GetName() == h_loaded.GetName()

    def test_roundtrip_3d_flow(self):
        h = ROOT.TH3D("h3f", "h3f", 2, 0, 2, 2, 0, 2, 2, 0, 2)
        h.Fill(-1, 0, 0)
        h.Fill(3, 3, 3)
        h[...] = np.arange(8).reshape(2, 2, 2)

        h_loaded = _roundtrip(h)

        assert np.array_equal(
            _bin_contents(h, flow=True),
            _bin_contents(h_loaded, flow=True),
        )

        assert h.GetName() == h_loaded.GetName()


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
