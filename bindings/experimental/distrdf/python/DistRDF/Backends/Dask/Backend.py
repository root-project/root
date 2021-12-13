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
import os

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF.Backends import Base
from DistRDF.Backends import Utils

try:
    import dask
    from dask.distributed import Client, LocalCluster, progress, get_worker
except ImportError:
    raise ImportError(("cannot import a Dask component. Refer to the Dask documentation "
                       "for installation instructions."))


class DaskBackend(Base.BaseBackend):
    """Dask backend for distributed RDataFrame."""

    def __init__(self, daskclient=None):
        super(DaskBackend, self).__init__()
        # If the user didn't explicitly pass a Client instance, the argument
        # `daskclient` will be `None`. In this case, we create a default Dask
        # client connected to a cluster instance with N worker processes, where
        # N is the number of cores on the local machine.
        self.client = (daskclient if daskclient is not None else
                       Client(LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1, processes=True)))

    def optimize_npartitions(self):
        """
        Attempts to compute a clever number of partitions for the current
        execution. Currently, we try to get the total number of worker logical
        cores in the cluster.
        """
        workers_dict = self.client.scheduler_info().get("workers")
        if workers_dict:
            # The 'workers' key exists in the dictionary and it is non-empty
            return sum(worker['nthreads'] for worker in workers_dict.values())
        else:
            # The scheduler doesn't have information about the workers
            return self.MIN_NPARTITIONS

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

        # These need to be passed as variables, because passing `self` inside
        # following `dask_mapper` function would trigger serialization errors
        # like the following:
        #
        # AttributeError: Can't pickle local object 'DaskBackend.ProcessAndMerge.<locals>.dask_mapper'
        # TypeError: cannot pickle '_asyncio.Task' object
        #
        # Which boil down to the self.client object not being serializable
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
                os.path.join(localdir, os.path.basename(filepath))
                for filepath in headers
            ]
            Utils.declare_headers(headers_on_executor)

            # Get and declare shared libraries on each worker
            shared_libs_on_ex = [
                os.path.join(localdir, os.path.basename(filepath))
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

        # Here we start the progressbar for the current RDF computation graph
        # running on the Dask client. This expects a future object, so we need
        # convert the last delayed object from the list above to a future
        # through the `persist` call. This also starts the computation in the
        # background, but the time difference is negligible. The progressbar is
        # properly shown in the terminal, whereas in the notebook it can be
        # shown only if it's the last call in a cell. Since we're encapsulating
        # it in this class, it won't be shown. Full details at
        # https://docs.dask.org/en/latest/diagnostics-distributed.html#dask.distributed.progress
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
