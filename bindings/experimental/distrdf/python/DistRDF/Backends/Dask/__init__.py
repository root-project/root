#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-11

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

def RDataFrame(*args, **kwargs):
    """
    Create an RDataFrame object that can run computations on a Dask cluster.
    """

    from DistRDF.Backends.Dask import Backend
    daskclient = kwargs.get("daskclient", None)
    daskbackend = Backend.DaskBackend(daskclient=daskclient)

    return daskbackend.make_dataframe(*args, **kwargs)
