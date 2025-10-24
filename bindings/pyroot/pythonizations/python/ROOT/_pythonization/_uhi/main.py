# Author: Silia Taider CERN  03/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import types
from typing import Any, Callable

from .indexing import _getitem, _iter, _setitem
from .plotting import _axes, _counts, _kind, _values_by_copy, _values_default, _variances
from .tags import _loc, _overflow, _rebin, _sum, _underflow

"""
Implementation of the module level helper functions for the UHI
"""


def _add_module_level_uhi_helpers(module: types.ModuleType) -> None:
    module.underflow = _underflow
    module.overflow = _overflow
    module.loc = _loc
    module.rebin = _rebin
    module.sum = _sum


"""
Implementation of the indexing component of the UHI
"""


def _add_indexing_features(klass: Any) -> None:
    klass.__getitem__ = _getitem
    klass.__setitem__ = _setitem
    klass.__iter__ = _iter


"""
Implementation of the plotting component of the UHI
"""


values_func_dict: dict[str, Callable] = {
    "TH1C": _values_by_copy,
    "TH2C": _values_by_copy,
    "TH3C": _values_by_copy,
    "TH2K": _values_by_copy,
    "TH3K": _values_by_copy,
    "TProfile": _values_by_copy,
    "TProfile2D": _values_by_copy,
    "TProfile2Poly": _values_by_copy,
    "TProfile3D": _values_by_copy,
}


def _add_plotting_features(klass: Any) -> None:
    klass.kind = property(_kind)
    klass.variances = _variances
    klass.counts = _counts
    klass.axes = property(_axes)
    klass.values = values_func_dict.get(klass.__name__, _values_default)
