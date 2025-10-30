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

import ROOT

from .plotting import PlottableAxisBase, _get_sum_of_weights, _get_sum_of_weights_squared, _hasWeights
from .tags import _get_axis

"""
Implementation of the serialization component of the UHI
"""


def _axis_to_dict(root_axis: ROOT.TAxis, uhi_axis: PlottableAxisBase) -> dict[str, Any]:
    return {
        "type": "regular",
        "lower": root_axis.GetBinLowEdge(root_axis.GetFirst()),
        "upper": root_axis.GetBinUpEdge(root_axis.GetLast()),
        "bins": root_axis.GetNbins(),
        "underflow": uhi_axis.traits.underflow,
        "overflow": uhi_axis.traits.overflow,
        "circular": uhi_axis.traits.circular,
    }


def _storage_to_dict(hist: Any) -> dict[str, Any]:
    """
    Logic:
    - If histogram is a profile (TProfile*) --> Kind="MEAN":
        - if histogram has Sumw2: type is weighted_mean_storage (if _hasWeights(hist))
        - else: storage type is mean_storage
    - Else (TH1*/TH2*/TH3*) --> Kind="COUNT":
        - if histogram has Sumw2: type is weighted_storage
        - else if histogram is TH*I: type is int_storage
        - else: type is double_storage
    """
    storage_dict = {
        "values": hist.values(),
    }

    if hist.kind == "MEAN":
        storage_dict["variances"] = hist.variances()

        if _hasWeights(hist):
            storage_dict["type"] = "weighted_mean"
            storage_dict["sum_of_weights"] = _get_sum_of_weights(hist)
            storage_dict["sum_of_weights_squared"] = _get_sum_of_weights_squared(hist)
        else:
            storage_dict["type"] = "mean"
            storage_dict["counts"] = hist.counts()

    else:  # COUNT
        if _hasWeights(hist):
            storage_dict["type"] = "weighted"
            storage_dict["variances"] = hist.variances()
        else:
            if hist.ClassName().endswith("I"):
                storage_dict["type"] = "int"
            else:
                storage_dict["type"] = "double"

    return storage_dict


def _to_uhi_(self) -> dict[str, Any]:
    return {
        "uhi_schema": 1,
        "writer_info": {"ROOT": {"version": ROOT.__version__, "class": self.ClassName()}},
        "axes": [_axis_to_dict(_get_axis(self, i), self.axes[i]) for i in range(self.GetDimension())],
        "storage": _storage_to_dict(self),
    }
