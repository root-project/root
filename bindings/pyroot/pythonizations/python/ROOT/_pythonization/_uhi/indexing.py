# Author: Silia Taider CERN  10/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

from contextlib import contextmanager

from .plotting import _values_by_copy
from .tags import _get_axis_len, _overflow, _rebin, _sum, _underflow

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


def _process_index_for_axis(self, index, axis, is_slice_stop=False):
    """Process an index for a histogram axis handling callables and index shifting."""
    if callable(index):
        # If the index is a `loc`, `underflow`, `overflow`, or `len`
        return _get_axis_len(self, axis) + 1 if index is len else index(self, axis)

    if isinstance(index, int):
        # -1 index returns the last valid bin
        if index == -1:
            return _overflow(self, axis) - 1

        # Shift the indices by 1 to align with the UHI convention,
        # where 0 corresponds to the first bin, unlike ROOT where 0 represents underflow and 1 is the first bin.
        nbins = _get_axis_len(self, axis) + (1 if is_slice_stop else 0)
        index = index + 1
        if abs(index) > nbins:
            raise IndexError(f"Histogram index {index - 1} out of range for axis {axis}. Valid range: (0,{nbins})")
        return index

    raise index


def _compute_uhi_index(self, index, axis, flow=True):
    """Convert tag functors to valid bin indices."""
    if isinstance(index, _rebin) or index is _sum:
        index = slice(None, None, index)

    if callable(index) or isinstance(index, int):
        return _process_index_for_axis(self, index, axis)

    if isinstance(index, slice):
        start, stop = _resolve_slice_indices(self, index, axis, flow)
        return slice(start, stop, index.step)

    raise TypeError(f"Unsupported index type: {type(index).__name__}")


def _compute_common_index(self, index, flow=True):
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
        ellipsis_pos = index.index(...)
        index = index[:ellipsis_pos] + (slice(None),) * (dim - len(index) + 1) + index[ellipsis_pos + 1 :]

    if len(index) != dim:
        raise IndexError(f"Expected {dim} indices, got {len(index)}")

    return [_compute_uhi_index(self, idx, axis, flow) for axis, idx in enumerate(index)]


def _setbin(self, index, value):
    """Set the bin content for a specific bin index"""
    self.SetBinContent(index, value)


def _resolve_slice_indices(self, index, axis, flow=True):
    """Resolve slice start and stop indices for a given axis"""
    start, stop = index.start, index.stop
    start = (
        _process_index_for_axis(self, start, axis) if start is not None else _underflow(self, axis) + (0 if flow else 1)
    )
    stop = (
        _process_index_for_axis(self, stop, axis, is_slice_stop=True)
        if stop is not None
        else _overflow(self, axis) + (1 if flow else 0)
    )
    if start < _underflow(self, axis) or stop > (_overflow(self, axis) + 1) or start > stop:
        raise IndexError(
            f"Slice indices {start, stop} out of range for axis {axis}. Valid range: {_underflow(self, axis), _overflow(self, axis) + 1}"
        )
    return start, stop


def _apply_actions(hist, actions, index, unprocessed_index, original_hist):
    """Apply rebinning or summing actions to the histogram, returns a new histogram"""
    if not actions or all(a is None for a in actions):
        return hist

    if any(a is _sum or a is sum for a in actions):
        sum_axes = tuple(i for i, a in enumerate(actions) if a is _sum or a is sum)
        if original_hist.GetDimension() == 1:
            # For the integral of a 1D histogram, we need special handling for the flow bins
            # h[::sum] is equivalent to h.Integral(0, nbins+1)
            # h[0:len:sum] is equivalent to h.Integral(1, nbins)
            start, stop = index[0].start, index[0].stop
            include_oflow = True if unprocessed_index.stop is None else False
            args = [start, stop - (1 if not include_oflow else 0)]
            hist = _sum(original_hist, sum_axes, args)
        else:
            hist = _sum(hist, sum_axes)

    if any(isinstance(a, _rebin) for a in actions):
        rebins = [a.ngroup if isinstance(a, _rebin) else 1 for a in actions if a is not _sum]
        hist = _rebin(rebins)(hist)

    if any(a is not None and not (isinstance(a, _rebin) or a is _sum or a is sum) for a in actions):
        raise ValueError(f"Unsupported action detected in actions {actions}")

    return hist


def _get_processed_slices(self, index):
    """Process slices and extract actions for each axis"""
    if len(index) != self.GetDimension():
        raise IndexError(f"Expected {self.GetDimension()} indices, got {len(index)}")
    processed_slices, actions = [], [None] * self.GetDimension()
    for axis, idx in enumerate(index):
        if isinstance(idx, slice):
            processed_slices.append((idx.start, idx.stop))
            actions[axis] = idx.step
        elif isinstance(idx, int):
            processed_slices.append((idx, idx + 1))
            actions[axis] = _sum
        else:
            raise TypeError(f"Unsupported index type: {type(idx).__name__}")

    return processed_slices, actions


def _slice_get(self, index, unprocessed_index):
    """
    This method creates a new histogram containing only the data from the
    specified slice.

    Steps:
    - Process the slices and extract the actions for each axis.
    - Get a new sliced histogram.
    - Apply any rebinning or summing actions to the resulting histogram.
    """
    import ROOT

    processed_slices, actions = _get_processed_slices(self, index)
    args_vec = ROOT.std.vector("Int_t")([item for pair in processed_slices for item in pair])

    target_hist = ROOT.Internal.Slice(self, args_vec)

    return _apply_actions(target_hist, actions, index, unprocessed_index, self)


def _slice_set(self, index, unprocessed_index, value):
    """
    This method modifies the histogram by updating the bin contents for the
    specified slice. It supports assigning a scalar value to all bins or
    assigning an array of values, provided the array's shape matches the slice.
    """
    import numpy as np

    import ROOT

    if not np.isscalar(value):
        try:
            value = np.asanyarray(value)
        except AttributeError:
            raise TypeError(f"Unsupported value type: {type(value).__name__}")

    # Depending on the shape of the array provided, we can set or not the flow bins
    # Setting with a scalar does not set the flow bins
    # broadcasting an array to the shape of the slice does not set the flow bins neither
    flow = False
    if isinstance(value, np.ndarray):
        processed_slices, _ = _get_processed_slices(self, index)
        slice_shape = tuple(stop - start for start, stop in processed_slices)
        flow = value.size == np.prod(slice_shape)

    if not flow:
        index = _compute_common_index(self, unprocessed_index, flow=False)

    processed_slices, actions = _get_processed_slices(self, index)
    slice_shape = tuple(stop - start for start, stop in processed_slices)
    slice_edges = ROOT.std.vector("std::pair<Int_t, Int_t>")()
    for start, stop in processed_slices:
        slice_edges.push_back(ROOT.std.pair("Int_t", "Int_t")(start, stop))

    if np.isscalar(value):
        value = ROOT.std.vector("Double_t")([value] * np.prod(slice_shape))
    else:
        if value.size != np.prod(slice_shape):
            try:
                value = np.broadcast_to(value, slice_shape)
            except ValueError:
                raise ValueError(f"Expected {np.prod(slice_shape)} bin values, got {value.size}")
        value = ROOT.std.vector("Double_t")(value.flatten().astype(np.float64))

    ROOT.Internal.SetSliceContent(self, value, slice_edges)

    _apply_actions(self, actions, index, unprocessed_index, self)


def _getitem(self, index):
    uhi_index = _compute_common_index(self, index)
    if all(isinstance(i, int) for i in uhi_index):
        return self.GetBinContent(*uhi_index)
    if any(isinstance(i, slice) for i in uhi_index):
        return _slice_get(self, uhi_index, index)


def _setitem(self, index, value):
    uhi_index = _compute_common_index(self, index)
    if all(isinstance(i, int) for i in uhi_index):
        _setbin(self, self.GetBin(*uhi_index), value)
    elif any(isinstance(i, slice) for i in uhi_index):
        _slice_set(self, uhi_index, index, value)


def _iter(self):
    array = _values_by_copy(self, flow=True)
    for val in array.flat:
        yield val.item()
