#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-11

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations
import warnings


def RDataFrame(*args, **kwargs):
    """
    Create an RDataFrame object that can run computations on a Dask cluster.
    """

    from DistRDF.Backends.Dask import Backend
    daskclient = kwargs.get("daskclient", None)
    executor = kwargs.get("executor", None)
    msg_warn = (
        "The keyword argument 'daskclient' is not necessary anymore and will "
        "be removed in a future release. To provide the distributed.Client "
        "object, use 'executor' instead."
    )
    msg_err = (
        "Both the 'daskclient' and 'executor' keyword arguments were provided. "
        "This is not supported. Please provide only the 'executor' argument."
    )

    if executor is not None and daskclient is not None:
        warnings.warn(msg_warn, FutureWarning)
        raise ValueError(msg_err)

    if daskclient is not None:
        warnings.warn(msg_warn, FutureWarning)
        executor = daskclient
        daskclient = None

    daskbackend = Backend.DaskBackend(daskclient=executor)

    return daskbackend.make_dataframe(*args, **kwargs)
