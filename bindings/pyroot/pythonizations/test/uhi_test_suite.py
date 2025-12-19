from __future__ import annotations

import numpy as np
import pytest
import ROOT
import uhi.testing.indexing
from ROOT._pythonization._uhi.indexing import _temporarily_disable_add_directory


def th1_eq(self, other):
    if self is other:
        return True
    return (
        isinstance(other, type(self))
        and self.kind == other.kind
        and np.array_equal(self.values(), other.values())
        and np.array_equal(self.counts(), other.counts())
        and np.array_equal(self.variances(), other.variances())
        and all(a == b for a, b in zip(self.axes, other.axes))
    )


@pytest.fixture(autouse=True)
def patch_th1_eq(monkeypatch):
    """
    Temporarily patch ROOT.TH1.__eq__ for tests
    Since __eq__ pythonizations are not defined, the comparison
    would fall back to object identity (`is`) and would fail
    """
    monkeypatch.setattr(ROOT.TH1, "__eq__", th1_eq)


class TestIndexing1D(uhi.testing.indexing.Indexing1D[ROOT.TH1D]):
    tag = ROOT.uhi

    @classmethod
    def make_histogram(cls) -> ROOT.TH1D:
        """
        Return a 1D histogram with 10 bins from 0 to 1. Each bin value increases
        by 2 compared to the previous one, starting at 0. The underflow bin is 3,
        and the overflow bin is 1.
        The histogram is constructed from the UHI schema returned by `get_uhi()`, a utility
        defined in the reference base class:
        https://github.com/scikit-hep/uhi/blob/7bf20edac71ba5fa648f6b33ccdd579df7f5e1a0/src/uhi/testing/indexing.py#L77
        """
        with _temporarily_disable_add_directory():
            return ROOT.TH1D(cls.get_uhi())


class TestIndexing2D(uhi.testing.indexing.Indexing2D[ROOT.TH2D]):
    tag = ROOT.uhi

    @classmethod
    def make_histogram(cls) -> ROOT.TH2D:
        """
        Return a 2D histogram with [2,5] bins. The contents are x+2y, where x and y
        are the bin indices.
        The histogram is constructed from the UHI schema returned by `get_uhi()`, a utility
        defined in the reference base class:
        https://github.com/scikit-hep/uhi/blob/7bf20edac71ba5fa648f6b33ccdd579df7f5e1a0/src/uhi/testing/indexing.py#L399
        """
        with _temporarily_disable_add_directory():
            return ROOT.TH2D(cls.get_uhi())


class TestIndexing3D(uhi.testing.indexing.Indexing3D[ROOT.TH3D]):
    tag = ROOT.uhi

    @classmethod
    def make_histogram(cls) -> ROOT.TH3D:
        """
         Return a 3D histogram with [2,5,10] bins. The contents are x+2y+3z, where x,
         y, and z are the bin indices.
         The histogram is constructed from the UHI schema returned by `get_uhi()`, a utility
         defined in the reference base class:
        https://github.com/scikit-hep/uhi/blob/7bf20edac71ba5fa648f6b33ccdd579df7f5e1a0/src/uhi/testing/indexing.py#L599
        """
        with _temporarily_disable_add_directory():
            return ROOT.TH3D(cls.get_uhi())


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
