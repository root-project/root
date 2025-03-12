# Author: Silia Taider CERN  03/2025

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations
from abc import ABC, abstractmethod
import enum
from typing import Tuple, Iterator, Union, Callable, Any


# Implementation of the plotting component of the UHI
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
    def traits(self) -> PlottableAxisTraits:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Union[Tuple[float, float], str]:
        pass

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

def _shape(hist: Any) -> Tuple[int, ...]:
    return tuple(getattr(hist, f"GetNbins{ax}")() + 2 for ax in "XYZ"[:hist.GetDimension()])

def _axes(self) -> Tuple[Union[PlottableAxisContinuous, PlottableAxisDiscrete], ...]:
    return tuple(
            PlottableAxisFactory.create(getattr(self, f"Get{ax}axis")()) for ax in "XYZ"[:self.GetDimension()]
        )

def _kind(self) -> Kind:
    return Kind.COUNT if not _hasWeights(self) else Kind.MEAN

def _values_default(self) -> np.typing.NDArray[Any]:
    import numpy as np

    llv = self.GetArray()
    ret = np.frombuffer(llv, dtype=llv.typecode, count=self.GetSize())
    return ret.reshape(_shape(self), order="F")[tuple([slice(1, -1)] * len(_shape(self)))]

# Special case for TH1K: we need the array length to correspond to the number of bins
# according to the UHI plotting protocol
def _values_by_copy(self) -> np.typing.NDArray[Any]:
    import numpy as np
    
    return np.array([self.GetBinContent(i) for i in range(1, self.GetNbinsX() + 1)])

def _variances(self) -> np.typing.NDArray[Any]:
    import numpy as np

    if not _hasWeights(self):
        return self.values()
    
    sumw2_array = self.GetSumw2()
    size = sumw2_array.GetSize()
    arr = np.frombuffer(sumw2_array.GetArray(), dtype=np.float64, count=size)

    reshaped = arr.reshape(_shape(self), order="F")
    return reshaped[tuple([slice(1, -1)] * len(_shape(self)))]

def _counts(self) -> np.typing.NDArray[Any]:
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
