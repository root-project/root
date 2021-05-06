# @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import ntpath  # Filename from path (should be platform-independent)

from DistRDF import DataFrame
from DistRDF import Node
from DistRDF.Backends import Base
from DistRDF.Backends import Utils

try:
    import dask
    from dask.distributed import Client, progress
except ImportError:
    raise ImportError(("cannot import a dask component. Refer to the Apache Spark documentation "
                       "for installation instructions."))


class DaskBackend(Base.BaseBackend):
    """Dask backend for distributed RDataFrame."""

    def __init__(self, daskclient=None):
        """Init function."""
        super(DaskBackend, self).__init__()

        self.client = daskclient if daskclient is not None else None

    def ProcessAndMerge(self, ranges, mapper, reducer):
        """
        Performs map-reduce using Dask framework.

        Args:
            mapper (function): A function that runs the computational graph
                and returns a list of values.

            reducer (function): A function that merges two lists that were
                returned by the mapper.

        Returns:
            list: A list representing the values of action nodes returned
            after computation (Map-Reduce).
        """
        # The Dask client has to be initialized inside some context and not on
        # global scope since it's using Python Multiprocessing and each process
        # fork needs independent environment (e.g. otherwise each process would
        # try recreating a connection to the Dask client).
        if self.client is None:
            self.client = Client()

        dmapper = dask.delayed(mapper)
        dreducer = dask.delayed(reducer)

        mergeables_lists = [dmapper(range) for range in ranges]

        while len(mergeables_lists) > 1:
            mergeables_lists.append(
                dreducer(mergeables_lists.pop(0), mergeables_lists.pop(0)))

        # Visualize dask graph
        # if self.config.get("visualize_dask_graph"):
        #     dask.visualize(mergeables_lists[0])

        final_results = mergeables_lists.pop().persist()

        progress(final_results)
        return final_results.compute()

    def optimize_npartitions(self, npartitions):
        """Optimize number of partitions if possible"""
        return npartitions

    def distribute_unique_paths(self, paths):
        """
        Spark supports sending files to the executors via the
        `SparkContext.addFile` method. This method receives in input the path
        to the file (relative to the path of the current python session). The
        file is initially added to the Spark driver and then sent to the
        workers when they are initialized.

        Args:
            paths (set): A set of paths to files that should be sent to the
                distributed workers.
        """
        pass

    def make_dataframe(self, *args, **kwargs):
        """Creates an instance of RDataFrame that can run on a Dask cluster."""
        headnode = Node.HeadNode(*args)
        return DataFrame.RDataFrame(headnode, self, **kwargs)
