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
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING, Union, Tuple
import math
import ROOT

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF.Backends import Base
from DistRDF.Backends import Utils

try:
    import dask
    from dask.distributed import Client, get_worker, LocalCluster, progress, as_completed
except ImportError:
    raise ImportError(("cannot import a Dask component. Refer to the Dask documentation "
                       "for installation instructions."))

if TYPE_CHECKING:
    from dask_jobqueue import JobQueueCluster
    from DistRDF import Ranges
    from DistRDF._graph_cache import ExecutionIdentifier


def get_total_cores_generic(client: Client) -> int:
    """
    Retrieve the total number of cores known to the Dask scheduler through the
    client connection.
    """
    return sum(client.ncores().values())


def get_total_cores_jobqueuecluster(cluster: JobQueueCluster) -> int:
    """
    Retrieve the total number of cores from a Dask cluster connected to some
    kind of batch system (HTCondor, Slurm...).
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


def get_total_cores(client: Client) -> int:
    """
    Retrieve the total number of cores of the Dask cluster.
    """
    try:
        # It may happen that the user is connected to a batch system. We try
        # to import the 'dask_jobqueue' module lazily to avoid a dependency.
        from dask_jobqueue import JobQueueCluster
        if isinstance(client.cluster, JobQueueCluster):
            return get_total_cores_jobqueuecluster(client.cluster)
    except ModuleNotFoundError:
        # We are not using 'dask_jobqueue', fall through to generic case
        pass

    return get_total_cores_generic(client)


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
        return get_total_cores(self.client)

    @staticmethod
    def dask_mapper(current_range: Tuple, 
                    headers: List[str], 
                    shared_libraries: List[str],
                    pcms: List[str],
                    files: List[str],
                    mapper: Callable) -> Callable:
        """
        Gets the paths to the file(s) in the current executor, then
        declares the headers found.

        Args:
            current_range (tuple): The current range of the dataset being
                processed on the executor.

            headers (list): List of header file paths.

            shared_libraries (list): List of shared library file paths.

            mapper (function): The map function to be executed on each executor.

        Returns:
            function: The map function to be executed on each executor,
            complete with all headers needed for the analysis.
        """
        # Retrieve the current worker local directory
        localdir = get_worker().local_directory
        
        #Get and declare headers on each worker
        headers_on_executor = [
            os.path.join(localdir, os.path.basename(filepath))
            for filepath in headers
        ]
        Utils.distribute_headers(headers_on_executor)

        # Get and declare shared libraries on each worker
        shared_libs_on_ex = [
            os.path.join(localdir, os.path.basename(filepath))
            for filepath in shared_libraries
        ]
                
        Utils.distribute_shared_libraries(shared_libs_on_ex)

        return mapper(current_range)

    def ProcessAndMerge(self,
                        ranges: List[Any],
                        mapper: Callable[[Ranges.DataRange,
                                        Callable[[Union[Ranges.EmptySourceRange, Ranges.TreeRangePerc]],
                                                    Base.TaskObjects],
                                        Callable[[ROOT.RDF.RNode, int], List],
                                        Callable],
                                        Base.TaskResult],
                        reducer: Callable[[Base.TaskResult, Base.TaskResult], Base.TaskResult],
                        ) -> Base.TaskResult:
        """
        Performs map-reduce using Dask framework.

        Args:
            ranges (list): A list of ranges to be processed.
            mapper (function): A function that runs the computational graph
                and returns a list of values.

            reducer (function): A function that merges two lists that were
                returned by the mapper.

        Returns:
            list: A list representing the values of action nodes returned
            after computation (Map-Reduce).
        """  
        self.distribute_unique_paths(self.headers) 
        self.distribute_unique_paths(self.shared_libraries)
        self.distribute_unique_paths(self.pcms)
        self.distribute_unique_paths(self.files)
        
        
        dmapper = dask.delayed(DaskBackend.dask_mapper)
        dreducer = dask.delayed(reducer)

        mergeables_lists = [dmapper(range, self.headers, self.shared_libraries, self.pcms, self.files, mapper) for range in ranges]
        
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

    def ProcessAndMergeLive(self,
                            ranges: List[Any],
                            mapper: Callable[[Ranges.DataRange,
                                            Callable[[Union[Ranges.EmptySourceRange, Ranges.TreeRangePerc]],
                                                    Base.TaskObjects],
                                            Callable[[ROOT.RDF.RNode, int], List],
                                            Callable],
                                            Base.TaskResult],
                            reducer: Callable[[Base.TaskResult, Base.TaskResult], Base.TaskResult],
                            drawables_info_dict: Dict[int, Tuple[List[Optional[Callable]], int, str]],
                            ) -> Base.TaskResult:
        """
        Performs real-time map-reduce using Dask framework, retrieving the partial results 
        as soon as they are available, allowing real-time data representation.

        Args:
            ranges (list): A list of ranges to be processed.

            mapper (function): A function that runs the computational graph
                and returns a list of values.

            reducer (function): A function that merges two lists that were
                returned by the mapper.

            drawables_info_dict (dict): A dictionary where keys are plot object IDs 
                and values are tuples containing optional callback functions, 
                index of the plot object, and operation name.

        Returns:
            merged_results (TaskResult): The merged result of the computation.
        """
        
        self.distribute_unique_paths(self.headers) 
        self.distribute_unique_paths(self.shared_libraries)
        self.distribute_unique_paths(self.pcms)
        self.distribute_unique_paths(self.files)
        
        
        # Set up Dask mapper
        dmapper = dask.delayed(DaskBackend.dask_mapper)
        mergeables_lists = [dmapper(range, self.headers, self.shared_libraries, self.pcms, self.files, mapper) for range in ranges]
        # Compute the delayed tasks to get Dask futures that can be passed to the as_completed method
        future_tasks = self.client.compute(mergeables_lists)

        # Save the current canvas
        backend_pad = ROOT.TVirtualPad.TContext()

        # Set up live visualization canvas
        c = self._setup_canvas(len(drawables_info_dict))

        # Process partial results and display plots
        merged_results = self._process_partial_results(c, drawables_info_dict, reducer, future_tasks)

        # Close the live visualization canvas canvas
        c.Close()    

        return merged_results
                
    def _setup_canvas(self, num_plots: int) -> ROOT.TCanvas:
        """
        Set up a TCanvas for live visualization with divided pads based on the number of plots.

        Args:
            num_plots (int): Number of plots to be displayed.

        Returns:
            c: The initialized TCanvas object.
        """
        # Define constants for canvas layout
        CANVAS_WIDTH = 800
        CANVAS_HEIGHT = 400

        canvas_rows = math.ceil(math.sqrt(num_plots))
        canvas_cols = math.ceil(num_plots / canvas_rows)
        c = ROOT.TCanvas("distrdf_backend", "distrdf_backend", CANVAS_WIDTH * canvas_rows, CANVAS_HEIGHT * canvas_cols)
        c.Divide(canvas_rows, canvas_cols)

        return c

    def _process_partial_results(self, 
                                canvas: ROOT.TCanvas, 
                                drawables_info_dict: Dict[int, Tuple[List[Optional[Callable]], int, str]],
                                reducer: Callable[[Base.TaskResult, Base.TaskResult], Base.TaskResult],
                                future_tasks: List[dask.Future]) -> Base.TaskResult:
        """
        Process partial results and display plots on the provided canvas.

        Args:
            canvas: The TCanvas object for displaying plots.
			
            drawables_info_dict (dict): A dictionary where keys are plot object IDs 
                and values are tuples containing optional callback functions, 
                index of the plot object, and operation name.
			
            reducer (function): A function for reducing partial results.
			
            future_tasks: Dask future tasks representing partial results.

        Returns:
            merged_results (TaskResult): The merged result of the computation.
        """
        merged_results: Base.TaskResult = None 
        cumulative_plots: Dict[int, Any] = {}

        # Collect all futures in batches that had arrived since the last iteration
        for batch in as_completed(future_tasks, with_results=True).batches():
            for future, result in batch:
               merged_results = reducer(merged_results, result) if merged_results else result
            
            mergeables = merged_results.mergeables
            
            for pad_num, (drawable_id, (callbacks_list, index, operation_name)) in enumerate(drawables_info_dict.items(), start=1):
                cumulative_plots[index] = mergeables[index].GetValue()

                pad = canvas.cd(pad_num)
                self._apply_callbacks_and_draw(pad, cumulative_plots, operation_name, index, callbacks_list)
        
        return merged_results

    def _apply_callbacks_and_draw(self,
                                  pad: ROOT.TPad,
                                  cumulative_plots: Dict[int, Any],
                                  operation_name: str,
                                  index: int,
                                  callbacks_list: List[Optional[Callable]]) -> None:
        """
        Apply callbacks and draw plots on the provided pad.

        Args:
            pad: The TPad object for drawing plots.
			
            cumulative_plots: A dictionary of the current merged partial results.
			
            callbacks_list: A list of callback functions to be applied.
			
            operation_name (str): Name of the operation associated with the plot.
		
            index (int): Index of the plot in cumulative_plots dictionary.
        """
        for callback in callbacks_list:
                if callback is not None:
                    callback(cumulative_plots[index])

        if operation_name in ["Graph", "GraphAsymmErrors"]:
            cumulative_plots[index].Draw("AP")
        else:
            cumulative_plots[index].Draw()

        pad.Update()

    def distribute_unique_paths(self, paths):
        """
        Dask supports sending files to the workers via the `Client.upload_file`
        method. Its stated purpose is to send local Python packages to the
        nodes, but in practice it uploads the file to the path stored in the
        `local_directory` attribute of each worker.
        """
        for filepath in paths:
            self.client.upload_file(filepath, load=False)

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

    def cleanup_cache(self, exec_id: ExecutionIdentifier) -> None:
        """
        Remove the computation graph identified by the input argument from the
        cache.
        """
        def remove_from_rdf_cache(exec_id: ExecutionIdentifier) -> None:
            from DistRDF._graph_cache import _ACTIONS_REGISTER, _RDF_REGISTER
            _ACTIONS_REGISTER.pop(exec_id, None)
            _RDF_REGISTER.pop(exec_id, None)

        return self.client.run(remove_from_rdf_cache, exec_id)
