#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import partial, singledispatch
from typing import Callable, Iterable, List, Optional, TYPE_CHECKING, Union

import ROOT

from DistRDF import Ranges
from DistRDF.Backends import Utils
from DistRDF.HeadNode import HeadNode, TaskObjects, TreeHeadNode

# Type hints only
if TYPE_CHECKING:
    from DistRDF.ComputationGraphGenerator import ComputationGraphGenerator


def setup_mapper(initialization_fn: Callable) -> None:
    """
    Perform initial setup steps common to every mapper function.
    """
    # Disable graphics functionality in ROOT. It is not needed inside a
    # distributed task
    ROOT.gROOT.SetBatch(True)
    # Enable thread safety for the whole mapper function. We need to do
    # this since two tasks could be invoking the C++ interpreter
    # simultaneously, given that this function will release the GIL
    # before calling into C++ to run the event loop. Dask multi-threaded
    # or even multi-process workers could trigger such a scenario.
    ROOT.EnableThreadSafety()

    # Run initialization method to prepare the worker runtime
    # environment
    initialization_fn()


def get_mergeable_values(starting_node: ROOT.RDF.RNode, range_id: int,
                         computation_graph_callable: Callable[[ROOT.RDF.RNode, int], List], optimized: bool) -> List:
    """
    Triggers the computation graph and returns a list of mergeable values.
    """
    if optimized:
        # Create the RDF computation graph and execute it on this ranged
        # dataset. The results of the actions of the graph and their types
        # are returned
        results, res_types = computation_graph_callable(starting_node, range_id)

        # Get RResultPtrs out of the type-erased RResultHandles by
        # instantiating with the type of the value
        mergeables = [
            ROOT.ROOT.Detail.RDF.GetMergeableValue(res.GetResultPtr[res_type]())
            if isinstance(res, ROOT.RDF.RResultHandle)
            else res
            for res, res_type in zip(results, res_types)
        ]
    else:
        # Output of the callable
        resultptr_list = computation_graph_callable(starting_node, range_id)

        mergeables = [Utils.get_mergeablevalue(resultptr) for resultptr in resultptr_list]

    return mergeables


@dataclass
class TaskResult:
    """
    Holds objects returned by a task in distributed execution.
    Attributes:
        mergeables: A list of the partial results of the mapper. Only in a
            TTree-based run, if the task has nothing to process then this
            attribute is None.
        entries_in_trees: A struct holding the amount of processed entries in
            the task, as well as a dictionary where each key is an identifier
            for a tree opened in the task and the value is the number of entries
            in that tree. This attribute is not None only in a TTree-based run.
    """
    mergeables: Optional[List]
    entries_in_trees: Optional[Ranges.TaskTreeEntries]


def distrdf_mapper(
        current_range: Union[Ranges.EmptySourceRange, Ranges.TreeRangePerc],
        build_rdf_from_range:  Callable[[Union[Ranges.EmptySourceRange, Ranges.TreeRangePerc]],
                                        TaskObjects],
        computation_graph_callable: Callable[[ROOT.RDF.RNode, int], List],
        initialization_fn: Callable,
        optimized: bool) -> TaskResult:
    """
    Maps the computation graph to the input logical range of entries.
    """
    setup_mapper(initialization_fn)

    # Build an RDataFrame instance for the current mapper task, based
    # on the type of the head node.
    rdf_plus = build_rdf_from_range(current_range)
    if rdf_plus.rdf is not None:
        mergeables = get_mergeable_values(rdf_plus.rdf, current_range.id, computation_graph_callable, optimized)
    else:
        mergeables = None

    return TaskResult(mergeables, rdf_plus.entries_in_trees)


def merge_values(mergeables_out: Iterable, mergeables_in: Iterable) -> Iterable:
    """
    Merge values of second argument into values of first argument and return
    first argument.
    """
    if mergeables_out is not None and mergeables_in is not None:

        for mergeable_out, mergeable_in in zip(mergeables_out, mergeables_in):
            Utils.merge_values(mergeable_out, mergeable_in)

    elif mergeables_out is None and mergeables_in is not None:
        mergeables_out = mergeables_in

    # This should treat the 4 possible cases:
    # 1. both arguments are non-empty: first if statement
    # 2. First argument is None and second is not empty: elif statement
    # 3. First argument is not empty and second is None: return first
    #    list, no need to do anything
    # 4. Both arguments are None: return first, it's None anyway.
    return mergeables_out


def distrdf_reducer(results_inout: TaskResult,
                    results_in: TaskResult) -> TaskResult:
    """
    Merges two given iterables of values that were returned by two mapper
    function executions. Returns the first argument with its values updated from
    the second.
    """
    mergeables_out, entries_in_trees_out = results_inout.mergeables, results_inout.entries_in_trees
    mergeables_in, entries_in_trees_in = results_in.mergeables, results_in.entries_in_trees

    if entries_in_trees_out is not None and entries_in_trees_in is not None:
        # Merge dictionaries of trees and their entries. Different tasks
        # might have to access the same tree, so we must not count its
        # entries more than once.
        entries_in_trees_out.trees_with_entries.update(entries_in_trees_in.trees_with_entries)
        # On the other hand, any two tasks will process different
        # entries, so we sum them
        entries_in_trees_out.processed_entries += entries_in_trees_in.processed_entries

    mergeables_updated = merge_values(mergeables_out, mergeables_in)

    return TaskResult(mergeables_updated, entries_in_trees_out)


@singledispatch
def handle_returned_values(headnode: HeadNode, values: TaskResult) -> Iterable:
    """
    Handle values returned after distributed execution. In general, nothing
    needs to be done here.
    """
    return values.mergeables


@handle_returned_values.register
def _(headnode: TreeHeadNode, values: TaskResult) -> Iterable:
    """
    Handle values returned after distributed execution. When the data source is
    a TTree, check that exactly the input files and all the entries in the
    dataset were processed during distributed execution.
    """
    if values.mergeables is None:
        raise RuntimeError("The distributed execution returned no values. "
                           "This can happen if all files in your dataset contain empty trees.")

    # User could have requested to read the same file multiple times indeed
    input_files_and_trees = [
        f"{filename}?#{treename}" for filename, treename in zip(headnode.inputfiles, headnode.subtreenames)
    ]
    files_counts = Counter(input_files_and_trees)

    entries_in_trees = values.entries_in_trees
    # Keys should be exactly the same
    if files_counts.keys() != entries_in_trees.trees_with_entries.keys():
        raise RuntimeError("The specified input files and the files that were "
                           "actually processed are not the same.")

    # Multiply the entries of each tree by the number of times it was
    # requested by the user
    for fullpath in files_counts:
        entries_in_trees.trees_with_entries[fullpath] *= files_counts[fullpath]

    total_dataset_entries = sum(entries_in_trees.trees_with_entries.values())
    if entries_in_trees.processed_entries != total_dataset_entries:
        raise RuntimeError(f"The dataset has {total_dataset_entries} entries, "
                           f"but {entries_in_trees.processed_entries} were processed.")

    return values.mergeables


class BaseBackend(ABC):
    """
    Base class for RDataFrame distributed backends.

    Attributes:
        supported_operations (list): List of operations supported by the
            backend.
        initialization (function): Store user's initialization method, if
            defined.
        headers (list): List of headers that need to be declared for the
            analysis.
        shared_libraries (list): List of shared libraries needed for the
            analysis.
    """

    initialization = staticmethod(lambda: None)

    headers = set()
    shared_libraries = set()

    # Define a minimum amount of partitions for any distributed RDataFrame.
    # This is a safe lower limit, to account for backends that may not support
    # the case where the distributed RDataFrame processes only one partition.
    MIN_NPARTITIONS = 2

    @classmethod
    def register_initialization(cls, fun, *args, **kwargs):
        """
        Convert the initialization function and its arguments into a callable
        without arguments. This callable is saved on the backend parent class.
        Therefore, changes on the runtime backend do not require users to set
        the initialization function again.

        Args:
            fun (function): Function to be executed.

            *args (list): Variable length argument list used to execute the
                function.

            **kwargs (dict): Keyword arguments used to execute the function.
        """
        cls.initialization = partial(fun, *args, **kwargs)
        fun(*args, **kwargs)

    def execute(self, generator: "ComputationGraphGenerator"):
        """
        Executes an RDataFrame computation graph on a distributed backend.

        Args:
            generator (ComputationGraphGenerator): A factory object for a
                computation graph. Its ``get_callable`` method will return a
                function responsible for creating the computation graph of a
                given RDataFrame object and a range of entries. The range is
                needed for the `Snapshot` operation.
        """
        # Check if the workflow must be generated in optimized mode
        optimized = ROOT.RDF.Experimental.Distributed.optimized

        if optimized:
            computation_graph_callable = generator.get_callable_optimized()
        else:
            computation_graph_callable = generator.get_callable()

        # Avoid having references to the instance inside the mapper
        initialization_fn = self.initialization

        # Build the ranges for the current dataset
        headnode = generator.headnode

        ranges = headnode.build_ranges()
        build_rdf_from_range = headnode.generate_rdf_creator()

        mapper = partial(distrdf_mapper,
                         build_rdf_from_range=build_rdf_from_range,
                         computation_graph_callable=computation_graph_callable,
                         initialization_fn=initialization_fn,
                         optimized=optimized)

        # Values produced after Map-Reduce
        returned_values = self.ProcessAndMerge(ranges, mapper, distrdf_reducer)

        # Extract actual results of the RDataFrame operations requested
        actual_values = handle_returned_values(headnode, returned_values)
        # List of action nodes in the same order as values
        nodes = headnode.get_action_nodes()

        # Set the value of every action node
        for node, value in zip(nodes, actual_values):
            Utils.set_value_on_node(value, node, self)

    @abstractmethod
    def ProcessAndMerge(self, ranges, mapper, reducer):
        """
        Subclasses must define how to run map-reduce functions on a given
        backend.
        """
        pass

    @abstractmethod
    def distribute_unique_paths(self, paths):
        """
        Subclasses must define how to send all files needed for the analysis
        (like headers and libraries) to the workers.
        """
        pass

    def optimize_npartitions(self):
        """
        Distributed backends may optimize the number of partitions of the
        current dataset or leave it as it is.
        """
        return self.MIN_NPARTITIONS

    def distribute_files(self, files_paths):
        """
        Sends to the workers the generic files needed by the user.

        Args:
            files_paths (str, iter): Paths to the files to be sent to the
                distributed workers.
        """
        files_to_distribute = set()

        if isinstance(files_paths, str):
            files_to_distribute.update(
                Utils.get_paths_set_from_string(files_paths))
        else:
            for path_string in files_paths:
                files_to_distribute.update(
                    Utils.get_paths_set_from_string(path_string))

        self.distribute_unique_paths(files_to_distribute)

    def distribute_headers(self, headers_paths):
        """
        Includes the C++ headers to be declared before execution.

        Args:
            headers_paths (str, iter): A string or an iterable (such as a
                list, set...) containing the paths to all necessary C++ headers
                as strings. This function accepts both paths to the headers
                themselves and paths to directories containing the headers.
        """
        headers_to_distribute = set()

        if isinstance(headers_paths, str):
            headers_to_distribute.update(
                Utils.get_paths_set_from_string(headers_paths))
        else:
            for path_string in headers_paths:
                headers_to_distribute.update(
                    Utils.get_paths_set_from_string(path_string))

        # Distribute header files to the workers
        self.distribute_unique_paths(headers_to_distribute)

        # Declare headers locally
        Utils.declare_headers(headers_to_distribute)

        # Finally, add everything to the includes set
        self.headers.update(headers_to_distribute)

    def distribute_shared_libraries(self, shared_libraries_paths):
        """
        Includes the C++ shared libraries to be declared before execution. If
        any pcm file is present in the same folder as the shared libraries, the
        function will try to retrieve them and distribute them.

        Args:
            shared_libraries_paths (str, iter): A string or an iterable (such as
                a list, set...) containing the paths to all necessary C++ shared
                libraries as strings. This function accepts both paths to the
                libraries themselves and paths to directories containing the
                libraries.
        """
        libraries_to_distribute = set()
        pcm_to_distribute = set()

        if isinstance(shared_libraries_paths, str):
            pcm_to_distribute, libraries_to_distribute = (
                Utils.check_pcm_in_library_path(shared_libraries_paths))
        else:
            for path_string in shared_libraries_paths:
                pcm, libraries = Utils.check_pcm_in_library_path(
                    path_string
                )
                libraries_to_distribute.update(libraries)
                pcm_to_distribute.update(pcm)

        # Distribute shared libraries and pcm files to the workers
        self.distribute_unique_paths(libraries_to_distribute)
        self.distribute_unique_paths(pcm_to_distribute)

        # Include shared libraries locally
        Utils.declare_shared_libraries(libraries_to_distribute)

        # Finally, add everything to the includes set
        self.shared_libraries.update(libraries_to_distribute)

    @abstractmethod
    def make_dataframe(self, *args, **kwargs):
        """
        Distributed backends have to take care of creating an RDataFrame object
        that can run distributedly.
        """
