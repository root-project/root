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
from functools import singledispatch
from typing import Any, Dict, Optional

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF.Backends import Base
from DistRDF.Backends import Utils

try:
    import dask
    from dask.distributed import Client, get_worker, LocalCluster, progress, SpecCluster
    from dask_jobqueue import JobQueueCluster
except ImportError:
    raise ImportError(("cannot import a Dask component. Refer to the Dask documentation "
                       "for installation instructions."))


@singledispatch
def get_total_cores(cluster: SpecCluster, client: Client) -> int:
    """
    Retrieve the total number of cores from a Dask cluster object.
    """
    # The Client.ncores() method returns the number of cores of each Dask
    # worker that is known to the scheduler
    return sum(client.ncores().values())


@get_total_cores.register
def _(cluster: JobQueueCluster, client: Client) -> int:
    """
    Retrieve the total number of cores from a Dask cluster object.
    """
    # Wrapping in a try-block in case any of the dictionaries do not have the
    # needed keys
    try:
        # In some cases the Dask scheduler doesn't know about available workers
        # at creation time. Most notably, when using batch systems like HTCondor
        # through dask-jobqueue, creating the cluster object doesn't actually
        # start the workers. The scheduler will know about available workers in
        # the cluster only after cluster.scale has been called and the resource
        # manager has granted the requested jobs. So at this point, we can only
        # rely on the information that was passed by the user as a specification
        # of the cluster object. This comes in the form:
        # {'WORKER-NAME-1': {'cls': <class 'dask.WORKERCLASS'>,
        #                    'options': {'CORES_OR_NTHREADS': N, ...}},
        #  'WORKER-NAME-2': {'cls': <class 'dask.WORKERCLASS'>,
        #                    'options': {'CORES_OR_NTHREADS': N, ...}}}
        # This concept can vary between different types of clusters, but in the
        # cluster types defined in dask-jobqueue the keys of the dictionary above
        # refer to the name of a job submission, which can then involve multiple
        # cores of a node.
        workers_spec: Dict[str, Any] = cluster.worker_spec
        # For each job, there is a sub-dictionary that contains the 'options'
        # key, which value is another dictionary with all the information
        # specified when creating the cluster object. This contains also the
        # 'cores' key for any type of dask-jobqueue cluster.
        return sum(spec["options"]["cores"] for spec in workers_spec.values())
    except KeyError as e:
        raise RuntimeError("Could not retrieve the provided worker specification from the Dask cluster object. "
                           "Please report this as a bug.") from e


class DaskBackend(Base.BaseBackend):
    """Dask backend for distributed RDataFrame."""

    def __init__(self, daskclient: Optional[Client] = None):
        super(DaskBackend, self).__init__()
        # If the user didn't explicitly pass a Client instance, the argument
        # `daskclient` will be `None`. In this case, we create a default Dask
        # client connected to a cluster instance with N worker processes, where
        # N is the number of cores on the local machine.
        self.client = (daskclient if daskclient is not None else
                       Client(LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1, processes=True)))

    def optimize_npartitions(self) -> int:
        """
        Attempts to compute a clever number of partitions for the current
        execution. Currently it is the number of cores of the Dask cluster,
        either retrieved if known or inferred from the user-provided cluster
        specification.
        """
        # We dispatch on the type of the cluster, but the API to retrieve the
        # number of available cores belongs to the client
        return get_total_cores(self.client.cluster, self.client)

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
        npartitions = kwargs.pop("npartitions", None)
        headnode = HeadNode.get_headnode(self, npartitions, *args)
        return DataFrame.RDataFrame(headnode)
