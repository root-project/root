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
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import hashlib
from typing import Callable, Iterable, List, Optional, TYPE_CHECKING, Union


import ROOT

from DistRDF import Ranges
from DistRDF.Backends import Utils

# Type hints only
if TYPE_CHECKING:
    from DistRDF._graph_cache import ExecutionIdentifier
    from DistRDF.HeadNode import TaskObjects
    from DistRDF.Ranges import DataRange


def setup_mapper(initialization_fn: Callable, code_to_declare: str) -> None:    
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
    
    # Declare all user code in one call
    ROOT.gInterpreter.Declare(code_to_declare)


def get_mergeable_values(starting_node: ROOT.RDF.RNode, range_id: int,
                         computation_graph_callable: Callable[[ROOT.RDF.RNode, int], List],
                         exec_id: ExecutionIdentifier) -> List:
    """
    Triggers the computation graph and returns a list of mergeable values.
    """

    actions = computation_graph_callable(starting_node, range_id, exec_id)

    mergeables = [Utils.get_mergeablevalue(action) for action in actions]

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
        current_range: Ranges.DataRange,
        build_rdf_from_range:  Callable[[Union[Ranges.EmptySourceRange, Ranges.TreeRangePerc]],
                                        TaskObjects],
        computation_graph_callable: Callable[[ROOT.RDF.RNode, int], List],
        initialization_fn: Callable,
        code_to_declare: str) -> TaskResult:
    """
    Maps the computation graph to the input logical range of entries.
    """
    # Wrap code that may be calling into C++ in a try-except block in order
    # to better propagate exceptions.
    try:
        setup_mapper(initialization_fn, code_to_declare)
        
        # Build an RDataFrame instance for the current mapper task, based
        # on the type of the head node.
        rdf_plus = build_rdf_from_range(current_range)
        if rdf_plus.rdf is not None:
            mergeables = get_mergeable_values(rdf_plus.rdf, current_range.id, computation_graph_callable,
                                              current_range.exec_id)
        else:
            mergeables = None
    except ROOT.std.exception as e:
        raise RuntimeError(f"C++ exception thrown:\n\t{type(e).__name__}: {e.what()}")

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

    # Wrap code that may be calling into C++ in a try-except block in order
    # to better propagate exceptions.
    try:
        mergeables_updated = merge_values(mergeables_out, mergeables_in)
    except ROOT.std.exception as e:
        raise RuntimeError(f"C++ exception thrown:\n\t{type(e).__name__}: {e.what()}")

    return TaskResult(mergeables_updated, entries_in_trees_out)


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
    files = set()
    pcms = set()
    shared_libraries = set()
    strings_to_declare = dict()

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

    @classmethod
    def register_declaration(cls, code_to_declare): 
        
        stripped = code_to_declare.strip()
        sha256 = hashlib.sha256()
        sha256.update(stripped.encode())
        hex = sha256.hexdigest()
        if cls.strings_to_declare.get(hex, None) is None:
            code_with_guard = f"#ifndef {hex}\n#define {hex}\n{stripped}\n#endif"
            cls.strings_to_declare[hex] = code_with_guard

        ROOT.gInterpreter.Declare(cls.strings_to_declare[hex])

    @classmethod
    def register_shared_lib(cls, paths_to_shared_libraries):
        
        libraries_to_distribute, pcms_to_distribute = Utils.register_shared_libs(paths_to_shared_libraries)
        
        cls.shared_libraries.update(libraries_to_distribute)
        cls.pcms.update(pcms_to_distribute)
    
    @classmethod
    def register_headers(cls, paths_to_headers):
        
        headers_to_distribute = Utils.register_headers(paths_to_headers)
        cls.headers.update(headers_to_distribute)
    
    @classmethod 
    def register_files(cls, paths_to_files):
        """
        Sends to the workers the generic files needed by the user.

        Args:
            files_paths (str, iter): Paths to the files to be sent to the
                distributed workers.
        """
        files_to_distribute = Utils.register_files(paths_to_files)
        cls.files.update(files_to_distribute)    
    
    @abstractmethod
    def ProcessAndMerge(self, ranges: List[DataRange],
                        mapper: Callable[..., TaskResult],
                        reducer: Callable[[TaskResult, TaskResult], TaskResult]) -> TaskResult:
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

    @abstractmethod
    def optimize_npartitions(self) -> int:
        """
        Return a default number of partitions to split the dataframe in,
        depending on the backend.
        """
        pass

    @abstractmethod
    def make_dataframe(self, *args, **kwargs):
        """
        Distributed backends have to take care of creating an RDataFrame object
        that can run distributedly.
        """

    def cleanup_cache(self, _: ExecutionIdentifier) -> None:
        """
        Remove the artifacts of the computation graph identified by the input
        argument.
        """
        pass
