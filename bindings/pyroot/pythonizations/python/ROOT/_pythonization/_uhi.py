# Author: Silia Taider CERN  03/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Tuple, Union

"""
Implementation of the module level helper functions for the UHI
"""


def _underflow(hist: Any, axis: int) -> int:
    return 0


def _overflow(hist: Any, axis: int) -> int:
    return _get_axis(hist, axis).GetNbins() + 1


class _loc:
    def __init__(self, *values: float) -> None:
        self.values = values
        self.offset = 0

    def __add__(self, other: int) -> _loc:
        if isinstance(other, int):
            self.offset += other
            return self
        raise TypeError("Unsupported type for addition")

    def __sub__(self, other: int) -> _loc:
        if isinstance(other, int):
            self.offset -= other
            return self
        raise TypeError("Unsupported type for subtraction")

    def __call__(self, hist: Any, axis: int) -> int:
        return _get_axis(hist, axis).FindBin(*self.values) + self.offset


class _rebin:
    def __init__(self, ngroup):
        self.ngroup = ngroup

    def __call__(self, hist):
        rebin_methods = {2: "Rebin2D", 3: "Rebin3D"}
        rebin_method = rebin_methods.get(hist.GetDimension(), "Rebin")
        rebin_method = getattr(hist, rebin_method)
        return rebin_method(*self.ngroup, newname=hist.GetName())


class _sum:
    def __call__(self, hist, axis):
        dim = hist.GetDimension()

        if isinstance(axis, int):
            axis = (axis,)
        if dim == 1:
            return hist.Integral()
        if dim == 2:
            return (
                hist.ProjectionX()
                if axis == (0,)
                else hist.ProjectionY()
                if axis == (1,)
                else self._invalid_axis(axis, dim)
            )
        if dim == 3:
            project_map = {
                (0,): "x",
                (1,): "y",
                (2,): "z",
                (0, 1): "xy",
                (1, 0): "yx",
                (0, 2): "xz",
                (2, 0): "zx",
                (1, 2): "yz",
                (2, 1): "zy",
            }
            return hist.Project3D(project_map[axis]) if axis in project_map else self._invalid_axis(axis, dim)
        raise NotImplementedError(f"Summing not implemented for {dim}D histograms")

    @staticmethod
    def _invalid_axis(axis, dim):
        raise ValueError(f"Invalid axis {axis} for {dim}D histogram")


def add_module_level_uhi_helpers(module: Any) -> None:
    module.underflow = _underflow
    module.overflow = _overflow
    module.loc = _loc
    module.rebin = _rebin
    module.sum = _sum


"""
Implementation of the indexing component of the UHI
"""


def _get_axis(self, axis):
    return getattr(self, f"Get{['X', 'Y', 'Z'][axis]}axis")()


def _compute_uhi_index(self, index, axis):
    """Convert tag functors to valid bin indices."""
    if isinstance(index, (_rebin, _sum)):
        index = slice(None, None, index)

    if callable(index):
        return index(self, axis)

    if isinstance(index, int):
        nbins = _get_axis(self, axis).GetNbins()
        if abs(index) >= nbins:
            raise IndexError(f"Histogram index {index} out of range for axis {axis}")
        return index

    if isinstance(index, slice):
        start, stop = _resolve_slice_indices(self, index, axis)
        return slice(start, stop, index.step)

    raise TypeError(f"Unsupported index type: {type(index).__name__}")


def _compute_common_index(self, index):
    """Normalize and expand the index to match the histogram dimension."""
    dim = self.GetDimension()
    if isinstance(index, dict):
        expanded_index = [slice(None)] * dim
        for axis, value in index.items():
            expanded_index[axis] = value
        index = tuple(expanded_index)

    if not isinstance(index, tuple):
        index = (index,)

    if index.count(...) > 1:
        raise IndexError("Only one ellipsis is allowed in the index.")

    if any(idx is ... for idx in index):
        expanded_index = []
        for idx in index:
            if idx is ...:
                break
            expanded_index.append(idx)
        # fill remaining dimensions with `slice(None)`
        expanded_index.extend([slice(None)] * (dim - len(expanded_index)))
        index = tuple(expanded_index)

    if len(index) != dim:
        raise IndexError(f"Expected {dim} indices, got {len(index)}")

    return [_compute_uhi_index(self, idx, axis) for axis, idx in enumerate(index)]


def _setbin(self, index, value):
    """Set the bin content for a specific bin index"""
    self.SetBinContent(index, value)


def _resolve_slice_indices(self, index, axis):
    """Resolve slice start and stop indices for a given axis"""
    start, stop = index.start, index.stop
    # exclude flow bins
    if start == 0 and stop is len:
        start = _underflow(self, axis) - 1
        stop = _overflow(self, axis)

    start = start(self, axis) if callable(start) else start or (_underflow(self, axis))
    stop = stop(self, axis) if callable(stop) else stop or (_overflow(self, axis) + 1)
    if start < _underflow(self, axis) or stop > (_overflow(self, axis) + 1) or start > stop:
        raise IndexError(f"Slice indices {start, stop} out of range for axis {axis}")
    return start, stop


def _apply_actions(hist, actions):
    """Apply rebinning or summing actions to the histogram, returns a new histogram"""
    if not actions or all(a is None for a in actions):
        return hist

    if any(isinstance(a, _rebin) for a in actions):
        rebins = [a.ngroup if isinstance(a, _rebin) else 1 for a in actions]
        hist = _rebin(rebins)(hist)

    if any(a is _sum for a in actions):
        sum_axes = tuple(i for i, a in enumerate(actions) if a is _sum)
        return _sum()(hist, sum_axes)

    if any(a is not None and not isinstance(a, (_rebin, _sum)) for a in actions):
        raise ValueError("Unsupported action detected in actions")

    return hist


def _get_processed_slices(self, index):
    """Process slices and extract actions for each axis"""
    processed_slices, actions = [], [None] * self.GetDimension()
    for i, idx in enumerate(index):
        if isinstance(idx, slice):
            processed_slices.append(range(idx.start, idx.stop))
            actions[i] = idx.step
        else:
            processed_slices.append([idx])

    return processed_slices, actions


def _get_slice_indices(slices):
    """Generate all combinations of slice indices"""
    import numpy as np

    return np.array(np.meshgrid(*slices)).T.reshape(-1, len(slices))


def _slice_get(self, index):
    """Retrieve a slice of the histogram based on the index, returns a new histogram"""
    processed_slices, actions = _get_processed_slices(self, index)
    slice_indices = _get_slice_indices(processed_slices)
    target_hist = self.Clone()
    target_hist.Reset()
    for indices in slice_indices:
        indices = tuple(map(int, indices))
        target_hist.SetBinContent(*indices, self.GetBinContent(self.GetBin(*indices)))
    target_hist.SetEntries(
        target_hist.GetEffectiveEntries()
        + sum(
            target_hist.GetBinContent(_underflow(target_hist, axis))
            + target_hist.GetBinContent(_overflow(target_hist, axis))
            for axis in range(self.GetDimension())
        )
    )

    return _apply_actions(target_hist, actions)


def _slice_set(self, index, value):
    """Set values for a slice of the histogram"""
    import numpy as np

    processed_slices, actions = _get_processed_slices(self, index)
    slice_indices = _get_slice_indices(processed_slices)
    if isinstance(value, np.ndarray):
        if value.size != len(slice_indices):
            raise ValueError(f"Shape mismatch: expected {len(slice_indices)} values, got {value.size}")

        for indices, val in zip(slice_indices, value.ravel()):
            _setbin(self, self.GetBin(*map(int, indices)), val)
    elif np.isscalar(value):
        for indices in slice_indices:
            _setbin(self, self.GetBin(*map(int, indices)), value)
    else:
        raise TypeError(f"Unsupported value type: {type(value).__name__}")

    _apply_actions(self, actions)


def _getitem(self, index):
    index = _compute_common_index(self, index)
    if all(isinstance(i, int) for i in index):
        return self.GetBinContent(*index)

    if any(isinstance(i, slice) for i in index):
        return _slice_get(self, index)


def _setitem(self, index, value):
    index = _compute_common_index(self, index)
    if all(isinstance(i, int) for i in index):
        _setbin(self, self.GetBin(*index), value)
    elif any(isinstance(i, slice) for i in index):
        _slice_set(self, index, value)


def add_indexing_features(klass: Any) -> None:
    klass.__getitem__ = _getitem
    klass.__setitem__ = _setitem


"""
Implementation of the plotting component of the UHI
"""


class Kind(str, enum.Enum):
    COUNT = "COUNT"
    MEAN = "MEAN"


class PlottableAxisTraits:
    def __init__(self, circular: bool = False, discrete: bool = False):
        self._circular = circular
        self._discrete = discrete

    @property
    def circular(self) -> bool:
        return self._circular

    @property
    def discrete(self) -> bool:
        return self._discrete


class PlottableAxisBase(ABC):
    def __init__(self, tAxis: Any) -> None:
        self.tAx = tAxis

    @property
    @abstractmethod
    def traits(self) -> PlottableAxisTraits: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Union[Tuple[float, float], str]: ...

    def __len__(self) -> int:
        return self.tAx.GetNbins()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PlottableAxisBase):
            return False
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    @abstractmethod
    def __iter__(self) -> Iterator[Union[Tuple[float, float], str]]:
        pass


class PlottableAxisContinuous(PlottableAxisBase):
    @property
    def traits(self) -> PlottableAxisTraits:
        return PlottableAxisTraits(circular=False, discrete=False)

    def __getitem__(self, index: int) -> Tuple[float, float]:
        return (self.tAx.GetBinLowEdge(index + 1), self.tAx.GetBinUpEdge(index + 1))

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        for i in range(len(self)):
            yield self[i]


class PlottableAxisDiscrete(PlottableAxisBase):
    @property
    def traits(self) -> PlottableAxisTraits:
        return PlottableAxisTraits(circular=False, discrete=True)

    def __getitem__(self, index: int) -> str:
        return self.tAx.GetBinLabel(index + 1)

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self[i]


class PlottableAxisFactory:
    @staticmethod
    def create(tAxis) -> Union[PlottableAxisContinuous, PlottableAxisDiscrete]:
        if all(tAxis.GetBinLabel(i + 1) for i in range(tAxis.GetNbins())):
            return PlottableAxisDiscrete(tAxis)
        return PlottableAxisContinuous(tAxis)


def _hasWeights(hist: Any) -> bool:
    return bool(hist.GetSumw2() and hist.GetSumw2N())


def _shape(hist: Any, include_flow_bins: bool = True) -> Tuple[int, ...]:
    return tuple(
        getattr(hist, f"GetNbins{ax}")() + (2 if include_flow_bins else 0) for ax in "XYZ"[: hist.GetDimension()]
    )


def _axes(self) -> Tuple[Union[PlottableAxisContinuous, PlottableAxisDiscrete], ...]:
    return tuple(PlottableAxisFactory.create(getattr(self, f"Get{ax}axis")()) for ax in "XYZ"[: self.GetDimension()])


def _kind(self) -> Kind:
    return Kind.COUNT if not _hasWeights(self) else Kind.MEAN


def _values_default(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    llv = self.GetArray()
    ret = np.frombuffer(llv, dtype=llv.typecode, count=self.GetSize())
    return ret.reshape(_shape(self), order="F")[tuple([slice(1, -1)] * len(_shape(self)))]


# Special case for TH1K: we need the array length to correspond to the number of bins
# according to the UHI plotting protocol
def _values_by_copy(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    return np.array([self.GetBinContent(i) for i in range(1, self.GetNbinsX() + 1)])


def _variances(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    if not _hasWeights(self):
        return self.values()

    sumw2_array = self.GetSumw2()
    size = sumw2_array.GetSize()
    arr = np.frombuffer(sumw2_array.GetArray(), dtype=np.float64, count=size)

    reshaped = arr.reshape(_shape(self), order="F")
    return reshaped[tuple([slice(1, -1)] * len(_shape(self)))]


def _counts(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    if not _hasWeights(self):
        return self.values()

    sumw = self.values()
    return np.divide(
        self.values() ** 2,
        self.variances(),
        out=np.zeros_like(sumw, dtype=np.float64),
        where=self.variances() != 0,
    )


values_func_dict: dict[str, Callable] = {
    "TH1C": _values_by_copy,
    "TH1K": _values_by_copy,
    "TProfile": _values_by_copy,
}


def add_plotting_features(klass: Any) -> None:
    klass.kind = property(_kind)
    klass.variances = _variances
    klass.counts = _counts
    klass.axes = property(_axes)
    klass.values = values_func_dict.get(klass.__name__, _values_default)
