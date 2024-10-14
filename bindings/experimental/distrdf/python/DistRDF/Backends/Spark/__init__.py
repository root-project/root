#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

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
    Create an RDataFrame object that can run computations on a Spark cluster.
    """

    from DistRDF.Backends.Spark import Backend
    sparkcontext = kwargs.get("sparkcontext", None)
    executor = kwargs.get("executor", None)
    msg_warn = (
        "The keyword argument 'sparkcontext' is not necessary anymore and will "
        "be removed in a future release. To provide the SparkContext object, "
        "use 'executor' instead."
    )
    msg_err = (
        "Both the 'sparkcontext' and 'executor' keyword arguments were provided. "
        "This is not supported. Please provide only the 'executor' argument."
    )

    if executor is not None and sparkcontext is not None:
        warnings.warn(msg_warn, FutureWarning)
        raise ValueError(msg_err)

    if sparkcontext is not None:
        warnings.warn(msg_warn, FutureWarning)
        executor = sparkcontext
        sparkcontext = None

    spark = Backend.SparkBackend(sparkcontext=executor)

    return spark.make_dataframe(*args, **kwargs)
