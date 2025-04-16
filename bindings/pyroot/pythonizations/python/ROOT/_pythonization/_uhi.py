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
import types
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Tuple, Union

"""
Implementation of the module level helper functions for the UHI
"""


def _underflow(hist: Any, axis: int) -> int:
    return 0


def _overflow(hist: Any, axis: int) -> int:
    return _get_axis(hist, axis).GetNbins() + 1


class _loc:
    """
    Represents a location-based index for histograms, returning the bin corresponding
    to a specified value on a given axis. Supports addition and subtraction to shift
    the computed bin by an integer offset.

    Example:
        v = h[loc(b) + 1]  # Returns the bin above the one containing the value `b`
    """

    def __init__(self, value: float) -> None:
        self.value = value
        self.offset = 0

    def __add__(self, other: int) -> _loc:
        if isinstance(other, int):
            self.offset += other
            return self
        raise TypeError(f"Unsupported type for addition: {type(other).__name__}. Expected an integer.")

    def __sub__(self, other: int) -> _loc:
        if isinstance(other, int):
            self.offset -= other
            return self
        raise TypeError(f"Unsupported type for substraction: {type(other).__name__}. Expected an integer.")

    def __call__(self, hist: Any, axis: int) -> int:
        return _get_axis(hist, axis).FindBin(self.value) + self.offset


class _rebin:
    """
    Represents a rebinning operation for histograms, where bins are grouped together
    by the factor ngroup.

    Example:
        h_rebinned = h[::ROOT.uhi.rebin(2)]  # Rebin the histogram with a grouping factor of 2
    """

    def __init__(self, ngroup):
        self.ngroup = ngroup

    def __call__(self, hist):
        rebin_methods = {1: "Rebin", 2: "Rebin2D", 3: "Rebin3D"}
        rebin_method = rebin_methods.get(hist.GetDimension())
        rebin_method = getattr(hist, rebin_method)
        return rebin_method(*self.ngroup, newname=hist.GetName())


def _sum(hist, axis):
    """
    Represents a summation operation for histograms, which either computes the integral
    (1D histograms) or projects the histogram along specified axes (2D and 3D histograms).

    Example:
        ans = h[::ROOT.uhi.sum]  # Compute the integral for a 1D histogram
        h_projected = h[:, ::ROOT.uhi.sum]  # Project the Y axis for a 2D histogram
        h_projected = h[:, :, ::ROOT.uhi.sum]  # Project the Z axis for a 3D histogram
    """
    dim = hist.GetDimension()

    def _invalid_axis(axis, dim):
        raise ValueError(f"Invalid axis {axis} for {dim}D histogram")

    if isinstance(axis, int):
        axis = (axis,)
    if dim == 1:
        return hist.Integral()
    if dim == 2:
        return hist.ProjectionX() if axis == (0,) else hist.ProjectionY() if axis == (1,) else _invalid_axis(axis, dim)
    if dim == 3:
        # It is not possible from the interface to specify the options "yx", "zy", "zx"
        project_map = {
            (0,): "yz",
            (1,): "xz",
            (2,): "xy",
            (0, 1): "z",
            (0, 2): "y",
            (1, 2): "x",
        }
        return hist.Project3D(project_map[axis]) if axis in project_map else _invalid_axis(axis, dim)
    raise NotImplementedError(f"Summing not implemented for {dim}D histograms")


def _add_module_level_uhi_helpers(module: types.ModuleType) -> None:
    module.underflow = _underflow
    module.overflow = _overflow
    module.loc = _loc
    module.rebin = _rebin
    module.sum = _sum


"""
Implementation of the indexing component of the UHI
"""


@contextmanager
def _temporarily_disable_add_directory():
    """
    Temporarily disable adding the new created histograms to the list of objects in memory
    """
    import ROOT

    old_status = ROOT.TH1.AddDirectoryStatus()
    ROOT.TH1.AddDirectory(False)
    try:
        yield
    finally:
        ROOT.TH1.AddDirectory(old_status)


def _get_axis(self, axis):
    return getattr(self, f"Get{['X', 'Y', 'Z'][axis]}axis")()


def _get_axis_len(self, axis):
    return _get_axis(self, axis).GetNbins()


def _process_index_for_axis(self, index, axis):
    """Process an index for a histogram axis handling callables and index shifting."""
    if callable(index):
        # If the index is a `loc`, `underflow`, `overflow`, or `len`
        return _get_axis_len(self, axis) if index is len else index(self, axis)

    if isinstance(index, int):
        # -1 index returns the last valid bin
        if index == -1:
            return _overflow(self, axis) - 1
        # Shift the indices by 1 to align with the UHI convention,
        # where 0 corresponds to the first bin, unlike ROOT where 0 represents underflow and 1 is the first bin.
        index = index + 1
        nbins = _get_axis_len(self, axis)
        if abs(index) > nbins:
            raise IndexError(f"Histogram index {index} out of range for axis {axis}")
        return index

    raise index


def _compute_uhi_index(self, index, axis, include_flow_bins=True):
    """Convert tag functors to valid bin indices."""
    if isinstance(index, _rebin) or index is _sum:
        index = slice(None, None, index)

    if callable(index) or isinstance(index, int):
        return _process_index_for_axis(self, index, axis)

    if isinstance(index, slice):
        start, stop = _resolve_slice_indices(self, index, axis, include_flow_bins)
        return slice(start, stop, index.step)

    raise TypeError(f"Unsupported index type: {type(index).__name__}")


def _compute_common_index(self, index, include_flow_bins=True):
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

    return [_compute_uhi_index(self, idx, axis, include_flow_bins) for axis, idx in enumerate(index)]


def _setbin(self, index, value):
    """Set the bin content for a specific bin index"""
    self.SetBinContent(index, value)


def _resolve_slice_indices(self, index, axis, include_flow_bins=True):
    """Resolve slice start and stop indices for a given axis"""
    start, stop = index.start, index.stop
    start = (
        _process_index_for_axis(self, start, axis)
        if start is not None
        else _underflow(self, axis) + (0 if include_flow_bins else 1)
    )
    stop = (
        _process_index_for_axis(self, stop, axis)
        if stop is not None
        else _overflow(self, axis) + (1 if include_flow_bins else 0)
    )
    if start < _underflow(self, axis) or stop > (_overflow(self, axis) + 1) or start > stop:
        raise IndexError(f"Slice indices {start, stop} out of range for axis {axis}")
    return start, stop


def _apply_actions(hist, actions):
    """Apply rebinning or summing actions to the histogram, returns a new histogram"""
    if not actions or all(a is None for a in actions):
        return hist

    if any(a is _sum for a in actions):
        sum_axes = tuple(i for i, a in enumerate(actions) if a is _sum)
        hist = _sum(hist, sum_axes)

    if any(isinstance(a, _rebin) for a in actions):
        rebins = [a.ngroup if isinstance(a, _rebin) else 1 for a in actions if a is not _sum]
        hist = _rebin(rebins)(hist)

    if any(a is not None and not (isinstance(a, _rebin) or a is _sum) for a in actions):
        raise ValueError(f"Unsupported action detected in actions {actions}")

    return hist


def _get_processed_slices(self, index):
    """Process slices and extract actions for each axis"""
    if len(index) != self.GetDimension():
        raise IndexError(f"Expected {self.GetDimension()} indices, got {len(index)}")
    processed_slices, out_of_range_indices, actions = [], [], [None] * self.GetDimension()
    for axis, idx in enumerate(index):
        axis_bins = range(_overflow(self, axis) + 1)
        if isinstance(idx, slice):
            slice_range = range(idx.start, idx.stop)
            processed_slices.append(slice_range)
            uflow = [b for b in axis_bins if b < idx.start]
            oflow = [b for b in axis_bins if b >= idx.stop]
            out_of_range_indices.append((uflow, oflow))
            actions[axis] = idx.step
        else:
            processed_slices.append([idx])

    return processed_slices, out_of_range_indices, actions


def _get_slice_indices(slices):
    """
    This function uses numpy's meshgrid to create a grid of indices from the input slices,
    and reshapes the grid into a list of all possible index combinations.

    Example:
        slices = [range(2), range(3)]
        # This represents two dimensions:
        #   - The first dimension has indices [0, 1]
        #   - The second dimension has indices [0, 1, 2]

        result = _get_slice_indices(slices)
        # result:
        # [[0, 0],
        #  [0, 1],
        #  [0, 2],
        #  [1, 0],
        #  [1, 1],
        #  [1, 2]]
    """
    import numpy as np

    grids = np.meshgrid(*slices, indexing="ij")
    return np.array(grids).reshape(len(slices), -1).T


def _set_flow_bins(self, target_hist, out_of_range_indices):
    """
    Accumulate content from bins outside the slice range into flow bins.
    """
    dim = self.GetDimension()
    uflow_bin = tuple(_underflow(self, axis) for axis in range(dim))
    oflow_bin = tuple(_overflow(self, axis) for axis in range(dim))
    flow_sum = 0

    for axis, (underflow_indices, overflow_indices) in enumerate(out_of_range_indices):
        all_axes = [range(_overflow(self, j)) for j in range(dim)]

        def sum_bin_content(indices_list, target_bin):
            current_val = target_hist.GetBinContent(*target_bin)
            temp_axes = list(all_axes)
            temp_axes[axis] = indices_list
            for idx in _get_slice_indices(temp_axes):
                current_val += self.GetBinContent(*tuple(map(int, idx)))
            target_hist.SetBinContent(*target_bin, current_val)
            return current_val

        flow_sum += sum_bin_content(underflow_indices, uflow_bin)
        flow_sum += sum_bin_content(overflow_indices, oflow_bin)

    return flow_sum


def _slice_get(self, index):
    """
    This method creates a new histogram containing only the data from the
    specified slice.

    Steps:
    - Process the slices and extract the actions for each axis.
    - Clone the original histogram and reset its contents.
    - Set the bin content for each index in the slice.
    - Update the number of entries in the cloned histogram (also updates the statistics).
    - Apply any rebinning or summing actions to the resulting histogram.
    """
    processed_slices, out_of_range_indices, actions = _get_processed_slices(self, index)
    slice_indices = _get_slice_indices(processed_slices)
    with _temporarily_disable_add_directory():
        target_hist = self.Clone()
        target_hist.Reset()

    for indices in slice_indices:
        indices = tuple(map(int, indices))
        target_hist.SetBinContent(*indices, self.GetBinContent(self.GetBin(*indices)))

    flow_sum = _set_flow_bins(self, target_hist, out_of_range_indices)

    target_hist.SetEntries(target_hist.GetEffectiveEntries() + flow_sum)

    return _apply_actions(target_hist, actions)


def _slice_set(self, index, unprocessed_index, value):
    """
    This method modifies the histogram by updating the bin contents for the
    specified slice. It supports assigning a scalar value to all bins or
    assigning an array of values, provided the array's shape matches the slice.
    """
    import numpy as np

    # Depending on the shape of the array provided, we can set or not the flow bins
    # Setting with a scalar does not set the flow bins
    include_flow_bins = not (
        (isinstance(value, np.ndarray) and value.shape == _shape(self, include_flow_bins=False)) or np.isscalar(value)
    )
    if not include_flow_bins:
        index = _compute_common_index(self, unprocessed_index, include_flow_bins=False)

    processed_slices, _, actions = _get_processed_slices(self, index)
    slice_indices = _get_slice_indices(processed_slices)
    if isinstance(value, np.ndarray):
        if value.size != len(slice_indices):
            raise ValueError(f"Expected {len(slice_indices)} bin values, got {value.size}")

        expected_shape = tuple(len(slice_range) for slice_range in processed_slices)
        if value.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {value.shape}")

        for indices, val in zip(slice_indices, value.ravel()):
            _setbin(self, self.GetBin(*map(int, indices)), val)
    elif np.isscalar(value):
        for indices in slice_indices:
            _setbin(self, self.GetBin(*map(int, indices)), value)
    else:
        raise TypeError(f"Unsupported value type: {type(value).__name__}")

    _apply_actions(self, actions)


def _getitem(self, index):
    uhi_index = _compute_common_index(self, index)
    if all(isinstance(i, int) for i in uhi_index):
        return self.GetBinContent(*uhi_index)

    if any(isinstance(i, slice) for i in uhi_index):
        return _slice_get(self, uhi_index)


def _setitem(self, index, value):
    uhi_index = _compute_common_index(self, index)
    if all(isinstance(i, int) for i in uhi_index):
        _setbin(self, self.GetBin(*uhi_index), value)
    elif any(isinstance(i, slice) for i in uhi_index):
        _slice_set(self, uhi_index, index, value)


def _eq(self, other):
    import numpy as np

    return (
        isinstance(other, type(self))
        and _shape(self) == _shape(other)
        and np.array_equal(_values_default(self), _values_default(other))
    )


def _add_indexing_features(klass: Any) -> None:
    klass.__getitem__ = _getitem
    klass.__setitem__ = _setitem
    klass.__eq__ = _eq


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
    return tuple(_get_axis(hist, i).GetNbins() + (2 if include_flow_bins else 0) for i in range(hist.GetDimension()))


def _axes(self) -> Tuple[Union[PlottableAxisContinuous, PlottableAxisDiscrete], ...]:
    return tuple(PlottableAxisFactory.create(_get_axis(self, i)) for i in range(self.GetDimension()))


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


def _add_plotting_features(klass: Any) -> None:
    klass.kind = property(_kind)
    klass.variances = _variances
    klass.counts = _counts
    klass.axes = property(_axes)
    klass.values = values_func_dict.get(klass.__name__, _values_default)
