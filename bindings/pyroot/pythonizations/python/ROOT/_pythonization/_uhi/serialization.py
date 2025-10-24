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

from .plotting import PlottableAxisBase

"""
Implementation of the serialization component of the UHI
"""


def _axis_to_dict(root_axis: ROOT.TAxis, uhi_axis: PlottableAxisBase) -> dict[str, Any]:
    """
    Return a dictionary representation of the given ROOT axis.
    """
    return {
        "type": "regular",
        "lower": root_axis.GetBinLowEdge(root_axis.GetFirst()),
        "upper": root_axis.GetBinUpEdge(root_axis.GetLast()),
        "bins": root_axis.GetNbins(),
        "underflow": uhi_axis.traits.underflow,
        "overflow": uhi_axis.traits.overflow,
        "circular": uhi_axis.traits.circular,
    }


def _axis_from_dict(axis_dict: dict[str, Any]) -> list[Any]:
    """
    Return the arguments needed to construct the corresponding ROOT histogram axis.
    For now only supports regular axes.
    """

    axis_type = axis_dict["type"]

    if axis_type == "regular":
        nbins = axis_dict["bins"]
        lower = axis_dict["lower"]
        upper = axis_dict["upper"]
        return [nbins, lower, upper]

    raise ValueError(f"Unsupported axis type for conversion to ROOT: {axis_type}")


def _storage_to_dict(hist: Any) -> dict[str, Any]:
    from .plotting import _get_sum_of_weights, _get_sum_of_weights_squared, _hasWeights

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
        "values": hist.values(flow=True),
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


def _generate_unique_hist_name(prefix="h_uhi", nhex=8) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:nhex]}"


def _get_ctor_args(self, uhi_dict: dict[str, Any]) -> ROOT.TH1:
    # rebuild axes
    axes = uhi_dict["axes"]
    axes_specs = [_axis_from_dict(axis_dict) for axis_dict in axes]

    # check if a name has been provided
    writer_info = uhi_dict.get("writer_info", {})
    root_info = writer_info.get("ROOT", {})
    name = root_info.get("name", None)
    if name is None:
        name = _generate_unique_hist_name()

    # constructor arguments
    ctor_args = [name, name]
    for axis_spec in axes_specs:
        ctor_args.extend(axis_spec)

    return ctor_args


def _set_histogram_storage_from_dict(hist: Any, storage_dict: dict[str, Any]) -> None:
    """
    Set the histogram storage (values and statistics) from the given storage dictionary.
    """
    hist_values = storage_dict["values"]
    hist[...] = hist_values


def _is_uhi_dict(obj: object) -> bool:
    return isinstance(obj, dict) and "uhi_schema" in obj


def _validate_uhi_dict_for_root_ctor(uhi_dict: dict[str, Any]) -> None:
    # Only support uhi_schema version 1 currently
    version = uhi_dict.get("uhi_schema")
    if version != 1:
        raise ValueError(f"Unsupported UHI schema version: {version}")

    # Minimal required keys for our implementation
    if "axes" not in uhi_dict or "storage" not in uhi_dict:
        raise ValueError("Invalid UHI dict: missing required keys 'axes' and 'storage'.")


def _to_uhi_(self) -> dict[str, Any]:
    from .tags import _get_axis

    return {
        "uhi_schema": 1,
        "writer_info": {
            "ROOT": {
                "version": ROOT.__version__,
                "class": self.ClassName(),
                "name": self.GetName(),
            }
        },
        "axes": [_axis_to_dict(_get_axis(self, i), self.axes[i]) for i in range(self.GetDimension())],
        "storage": _storage_to_dict(self),
    }


def _from_uhi_(cls, uhi_dict: dict[str, Any]) -> ROOT.TH1:
    # validate input
    if not _is_uhi_dict(uhi_dict):
        raise ValueError("Input is not a valid UHI dictionary.")

    _validate_uhi_dict_for_root_ctor(uhi_dict)

    from .indexing import _temporarily_disable_add_directory

    # get constructor arguments
    ctor_args = _get_ctor_args(cls, uhi_dict)

    # allocate instance
    self = cls.__new__(cls)

    # call original constructor
    # don't add to the global directory
    with _temporarily_disable_add_directory():
        self._original_init_(*ctor_args)

    # set storage
    _set_histogram_storage_from_dict(self, uhi_dict["storage"])

    return self
