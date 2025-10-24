# Author: Silia Taider CERN  10/2025

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
from typing import Any, Iterator, Tuple, Union

from .tags import _get_axis, _get_axis_len

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

    @property
    def underflow(self) -> bool:
        return True

    @property
    def overflow(self) -> bool:
        return True


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


def _axes(self) -> Tuple[Union[PlottableAxisContinuous, PlottableAxisDiscrete], ...]:
    return tuple(PlottableAxisFactory.create(_get_axis(self, i)) for i in range(self.GetDimension()))

# TODO this is not correct?
def _kind(self) -> Kind:
    return Kind.COUNT if not _hasWeights(self) else Kind.MEAN


def _shape(hist: Any, include_flow_bins: bool = True) -> Tuple[int, ...]:
    return tuple(_get_axis_len(hist, i, include_flow_bins) for i in range(hist.GetDimension()))


def _values_default(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    llv = self.GetArray()
    ret = np.frombuffer(llv, dtype=llv.typecode, count=self.GetSize())
    return ret.reshape(_shape(self), order="F")[tuple([slice(1, -1)] * len(_shape(self)))]


def _values_by_copy(self, include_flow_bins=False) -> np.typing.NDArray[Any]:  # noqa: F821
    from itertools import product

    import numpy as np

    offset = 0 if include_flow_bins else 1
    dimensions = [
        range(offset, _get_axis_len(self, axis, include_flow_bins=include_flow_bins) + offset)
        for axis in range(self.GetDimension())
    ]
    bin_combinations = product(*dimensions)

    return np.array([self.GetBinContent(*bin) for bin in bin_combinations]).reshape(
        _shape(self, include_flow_bins=include_flow_bins)
    )


def _variances(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    sum_of_weights = self.values()

    if not _hasWeights(self) and _kind(self) == Kind.COUNT:
        return sum_of_weights

    sum_of_weights_squared = _get_sum_of_weights_squared(self)

    if _kind(self) == Kind.MEAN:
        counts = self.counts()
        variances = sum_of_weights_squared.copy()
        variances[counts <= 1] = np.nan
        return variances

    return sum_of_weights_squared


def _counts(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    sum_of_weights = self.values()

    if not _hasWeights(self):
        return sum_of_weights

    sum_of_weights_squared = _get_sum_of_weights_squared(self)

    return np.divide(
        sum_of_weights**2,
        sum_of_weights_squared,
        out=np.zeros_like(sum_of_weights, dtype=np.float64),
        where=sum_of_weights_squared != 0,
    )

def _get_sum_of_weights(self) -> np.typing.NDArray[Any]:  # noqa: F821
    return self.values()

def _get_sum_of_weights_squared(self) -> np.typing.NDArray[Any]:  # noqa: F821
    import numpy as np

    shape = _shape(self, include_flow_bins=False)
    sumw2_arr = np.frombuffer(
        self.GetSumw2().GetArray(),
        dtype=self.GetSumw2().GetArray().typecode,
        count=self.GetSumw2().GetSize(),
    )
    return sumw2_arr[tuple([slice(1, -1)] * len(shape))].reshape(shape, order="F") if sumw2_arr.size > 0 else sumw2_arr
