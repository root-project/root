## @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

def RDataFrame(*args, **kwargs):
    """
    Create an RDataFrame object that can run computations on a Spark cluster.
    """

    from DistRDF.Backends.Spark import Backend
    sparkcontext = kwargs.get("sparkcontext", None)
    spark = Backend.SparkBackend(sparkcontext=sparkcontext)

    return spark.make_dataframe(*args, **kwargs)
