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
import os

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF.Backends import Base
from DistRDF.Backends import Utils

try:
    import dask
    from dask.distributed import Client, progress, get_worker
except ImportError:
    raise ImportError(("cannot import a Dask component. Refer to the Dask documentation "
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

        # With the Spark backend these need to be passed as variables.
        # Doing the same here for coherency, might change in the future.
        headers = self.headers
        shared_libraries = self.shared_libraries

        def dask_mapper(current_range):
            """
            Gets the paths to the file(s) in the current executor, then
            declares the headers found.

            Args:
                current_range (tuple): The current range of the dataset being
                    processed on the executor.

            Returns:
                function: The map function to be executed on each executor,
                complete with all headers needed for the analysis.
            """
            # Retrieve the current worker local directory
            localdir = get_worker().local_directory

            # Get and declare headers on each worker
            headers_on_executor = [
                os.path.join(localdir, filepath)
                for filepath in headers
            ]
            Utils.declare_headers(headers_on_executor)

            # Get and declare shared libraries on each worker
            shared_libs_on_ex = [
                os.path.join(localdir, filepath)
                for filepath in shared_libraries
            ]
            Utils.declare_shared_libraries(shared_libs_on_ex)

            return mapper(current_range)

        dmapper = dask.delayed(dask_mapper)
        dreducer = dask.delayed(reducer)

        mergeables_lists = [dmapper(range) for range in ranges]

        while len(mergeables_lists) > 1:
            mergeables_lists.append(
                dreducer(mergeables_lists.pop(0), mergeables_lists.pop(0)))

        final_results = mergeables_lists.pop().persist()

        progress(final_results)
        return final_results.compute()

    def distribute_unique_paths(self, paths):
        """
        Dask supports sending files to the workes via the `Client.upload_file`
        method. Its stated purpose is to send local Python packages to the
        nodes, but in practice it uploads the file to the path stored in the
        `local_directory` attribute of each worker.
        """
        for filepath in paths:
            self.client.upload_file(filepath)

    def make_dataframe(self, *args, **kwargs):
        """
        Creates an instance of distributed RDataFrame that can send computations
        to a Dask cluster.
        """
        # Set the number of partitions for this dataframe, one of the following:
        # 1. User-supplied `npartitions` optional argument
        # 2. An educated guess according to the backend, using the backend's
        #    `optimize_npartitions` function
        # 3. Set `npartitions` to 2
        npartitions = kwargs.pop("npartitions", self.optimize_npartitions())
        headnode = HeadNode.get_headnode(npartitions, *args)
        return DataFrame.RDataFrame(headnode, self)
