# Author: Silia Taider CERN  10/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

from typing import Any


def _get_axis(self, axis):
    return getattr(self, f"Get{['X', 'Y', 'Z'][axis]}axis")()


def _get_axis_len(self, axis, flow=False):
    return _get_axis(self, axis).GetNbins() + (2 if flow else 0)


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


def _sum(hist, axis, args=None):
    """
    Represents a summation operation for histograms, which either computes the integral (1D histograms)
    or projects the histogram along specified axes (projection is only for 2D and 3D histograms).

    Example:
        ans = h[0:len:ROOT.uhi.sum]  # Compute the integral for a 1D histogram excluding flow bins
        ans_2 = h[::ROOT.uhi.sum, ::ROOT.uhi.sum]  # Compute the integral for a 2D histogram including flow bins
        h_projected = h[:, ::ROOT.uhi.sum]  # Project the Y axis for a 2D histogram
        h_projected = h[:, :, ::ROOT.uhi.sum]  # Project the Z axis for a 3D histogram
    """
    dim = hist.GetDimension()

    def _invalid_axis(axis, dim):
        raise ValueError(f"Invalid axis {axis} for {dim}D histogram")

    if isinstance(axis, int):
        axis = (axis,)
    if dim == 1:
        return hist.Integral(*args) if axis == (0,) else _invalid_axis(axis, dim)
    if dim == 2:
        if axis == (0,):
            return hist.ProjectionY()
        elif axis == (1,):
            return hist.ProjectionX()
        elif axis == (0, 1):
            return hist.Integral()
        else:
            return _invalid_axis(axis, dim)
    if dim == 3:
        # It is not possible from the interface to specify the options "xy", "yz", "xz"
        project_map = {
            (0,): "zy",
            (1,): "zx",
            (2,): "yx",
            (0, 1): "z",
            (0, 2): "y",
            (1, 2): "x",
        }
        if axis == (0, 1, 2):
            return hist.Integral()
        return hist.Project3D(project_map[axis]) if axis in project_map else _invalid_axis(axis, dim)
    raise NotImplementedError(f"Summing not implemented for {dim}D histograms")
