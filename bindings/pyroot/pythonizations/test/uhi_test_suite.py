from __future__ import annotations

import numpy as np
import pytest
import ROOT
import uhi.testing.indexing
from ROOT._pythonization._uhi import _temporarily_disable_add_directory


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
        Reference base class:
        https://github.com/scikit-hep/uhi/blob/7d2cc3c169b51b5538e1380e66b892f451c28c1a/src/uhi/testing/indexing.py#L68
        """
        with _temporarily_disable_add_directory():
            h = ROOT.TH1D("h", "h", 10, 0, 1)
            h[...] = np.array([3.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 1.0])
            return h


class TestIndexing2D(uhi.testing.indexing.Indexing2D[ROOT.TH2D]):
    tag = ROOT.uhi

    @classmethod
    def make_histogram(cls) -> ROOT.TH2D:
        """
        Return a 2D histogram with [2,5] bins. The contents are x+2y, where x and y
        are the bin indices.
        Reference base class:
        https://github.com/scikit-hep/uhi/blob/7d2cc3c169b51b5538e1380e66b892f451c28c1a/src/uhi/testing/indexing.py#L391
        """
        with _temporarily_disable_add_directory():
            h2 = ROOT.TH2D("h2", "h2", 2, 0, 2, 5, 0, 5)
            x, y = np.mgrid[0:2, 0:5]
            data = np.pad(x + 2 * y, 1, mode="constant")
            h2[...] = data
            return h2


class TestIndexing3D(uhi.testing.indexing.Indexing3D[ROOT.TH3D]):
    tag = ROOT.uhi

    @classmethod
    def make_histogram(cls) -> ROOT.TH3D:
        """
        Return a 3D histogram with [2,5,10] bins. The contents are x+2y+3z, where x,
        y, and z are the bin indices.
        Reference base class:
        https://github.com/scikit-hep/uhi/blob/7d2cc3c169b51b5538e1380e66b892f451c28c1a/src/uhi/testing/indexing.py#L591
        """
        with _temporarily_disable_add_directory():
            h3 = ROOT.TH3D("h3", "h3", 2, 0, 2, 5, 0, 5, 10, 0, 10)
            x, y, z = np.mgrid[0:2, 0:5, 0:10]
            data = np.pad(x + 2 * y + 3 * z, 1, mode="constant")
            h3[...] = data
            return h3


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
