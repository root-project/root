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

import functools
from abc import ABCMeta
from abc import abstractmethod

import ROOT

from DistRDF.Backends import Utils

# Abstract class declaration
# This ensures compatibility between Python 2 and 3 versions, since in
# Python 2 there is no ABC class
ABC = ABCMeta("ABC", (object,), {})


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

    supported_operations = [
        "AsNumpy",
        "Count",
        "Define",
        "DefinePerSample",
        "Fill",
        "Filter",
        "Graph",
        "Histo1D",
        "Histo2D",
        "Histo3D",
        "HistoND",
        "Max",
        "Mean",
        "Min",
        "Profile1D",
        "Profile2D",
        "Profile3D",
        "Redefine",
        "Snapshot",
        "Sum"
    ]

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
        cls.initialization = functools.partial(fun, *args, **kwargs)
        fun(*args, **kwargs)

    def check_supported(self, operation_name):
        """
        Checks if a given operation is supported
        by the given backend.

        Args:
            operation_name (str): Name of the operation to be checked.

        Raises:
            Exception: This happens when `operation_name` doesn't exist
            the `supported_operations` instance attribute.
        """
        if operation_name not in self.supported_operations:
            raise Exception(
                "The current backend doesn't support \"{}\" !"
                .format(operation_name)
            )

    def execute(self, generator):
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
        initialization = self.initialization

        # Build the ranges for the current dataset
        headnode = generator.headnode
        ranges = headnode.build_ranges()
        build_rdf_from_range = headnode.generate_rdf_creator()

        def mapper(current_range):
            """
            Triggers the event-loop and executes all
            nodes in the computational graph using the
            callable.

            Args:
                current_range (Range): A Range named tuple, representing the
                    range of entries to be processed, their input files and
                    information about friend trees.

            Returns:
                list: This respresents the list of (mergeable)values of all
                action nodes in the computational graph.
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

            # We have to decide whether to do this in Dist or in subclasses
            # Utils.declare_headers(worker_includes)  # Declare headers if any
            # Run initialization method to prepare the worker runtime
            # environment
            initialization()

            # Build an RDataFrame instance for the current mapper task, based
            # on the type of the head node.
            rdf = build_rdf_from_range(current_range)

            if optimized:
                # Create the RDF computation graph and execute it on this ranged
                # dataset. The results of the actions of the graph and their types
                # are returned
                results, res_types = computation_graph_callable(rdf, current_range.id)

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
                resultptr_list = computation_graph_callable(rdf, current_range.id)

                mergeables = [Utils.get_mergeablevalue(resultptr) for resultptr in resultptr_list]

            return mergeables

        def reducer(mergeables_out, mergeables_in):
            """
            Merges two given lists of values that were
            returned by the mapper function for two different
            ranges.

            Args:
                mergeables_out (list): A list of computed (mergeable)values for
                    a given entry range in a dataset. The elements of this list
                    will be updated with the information contained in the
                    elements of the other argument list.

                mergeables_in (list): A list of computed (mergeable)values for
                    a given entry range in a dataset.

            Returns:
                list: The list of updated (mergeable)values.
            """

            for mergeable_out, mergeable_in in zip(mergeables_out, mergeables_in):
                Utils.merge_values(mergeable_out, mergeable_in)

            return mergeables_out

        # Values produced after Map-Reduce
        values = self.ProcessAndMerge(ranges, mapper, reducer)
        # List of action nodes in the same order as values
        nodes = generator.get_action_nodes()

        # Set the value of every action node
        for node, value in zip(nodes, values):
            if node.operation.name == "Snapshot":
                # Retrieving a new distributed RDataFrame from the result of a
                # distributed Snapshot needs knowledge of the correct backend
                node.value = value.GetValue(self)
            else:
                node.value = value.GetValue()

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
