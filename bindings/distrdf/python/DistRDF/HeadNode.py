from __future__ import annotations

import logging
import pickle
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, singledispatch
from itertools import zip_longest
from typing import TYPE_CHECKING, Callable, Deque, Dict, Iterable, List, Optional, Union

import ROOT

from DistRDF import ComputationGraphGenerator, Ranges, _graph_cache
from DistRDF.Backends import Utils
from DistRDF.Backends.Base import distrdf_mapper, distrdf_reducer
from DistRDF.Node import Node
from DistRDF.Operation import Action, InstantAction, Operation

if TYPE_CHECKING:
    from DistRDF.Backends.Base import BaseBackend, TaskResult

logger = logging.getLogger(__name__)


@singledispatch
def _append_node_to_actions(operation: Operation, node: Node, actions: List[Node]) -> None:
    """
    Appends the input node to the list of action nodes, if the operation of the
    node is an action.
    """
    pass


@_append_node_to_actions.register(Action)
@_append_node_to_actions.register(InstantAction)
def _(operation: Union[Action, InstantAction], node: Node, actions: List[Node]) -> None:
    actions.append(node)


@dataclass
class TaskObjects:
    """
    Holds objects needed in a distributed task.
    Attributes:
        rdf: The starting node of the RDataFrame computation graph. Only in a
            TTree-based run, if the task has nothing to process then this
            attribute is None.
        entries_in_trees: A struct holding the amount of processed entries in
            the task, as well as a dictionary where each key is an identifier
            for a tree opened in the task and the value is the number of entries
            in that tree. This attribute is not None only in a TTree-based run.
    """
    rdf: Optional[ROOT.RDF.RNode]
    entries_in_trees: Optional[Ranges.TaskTreeEntries]


class HeadNode(Node, ABC):
    """
    The head node of the computation graph. Keeps record of all nodes in the
    graph, is then able to generate a flat representation to be sent to the
    distributed workers.

    Attributes:
        backend: A reference to the instance of distributed backend that will
            execute this computation graph.

        graph_nodes: A deque of references to the nodes belonging to this graph.

        node_counter: A counter of how many nodes were created in the graph of
            this head node, starting from zero.
    """

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], localdf: ROOT.RDataFrame):
        super().__init__(lambda: self)

        self.backend = backend

        self.node_counter: int = 0
        # It is important to have a double-ended queue because we need to
        # traverse the nodes in different ways w.r.t. their insertion order.
        # While pruning the graph, we need to check leaf nodes before their
        # parents, so if a child is pruned then it also decrements the counter
        # of children of its parent. Thus, we need a bottom-up traversal.
        # While executing the graph in a task, we need to create the RDF nodes
        # in the order the user requested them, e.g. starting from the
        # RDataFrame itself, then calling its direct children, their children
        # and so on. Thus, we need a top-down traversal.
        self.graph_nodes: Deque[Node] = deque([self])

        # Uniquely identify each computation graph execution of this RDataFrame
        # The uuid can be set here
        self.rdf_uuid = uuid.uuid4()
        # The full identifier is created at the beginning of each execution
        self.exec_id: _graph_cache.ExecutionIdentifier = None

        # Internal attribute to keep track of the number of partitions. We also
        # check whether it was specified by the user when creating the dataframe.
        # If so, this attribute will not be updated when triggering.
        self._npartitions = npartitions
        self._user_specified_npartitions = True if npartitions is not None else False

        # Internal RDataFrame object, useful to expose information such as
        # column names.
        self.rdf_node = localdf

        # A dictionary where the keys are the IDs of the objects to live visualize
        # and the values are the corresponding callback functions 
        # This attribute only gets set in case the LiveVisualize() function is called
        self.drawables_dict: Optional[Dict[int, List[Optional[Callable]]]] = None

    def __del__(self):
        """
        Remove the reference to the local RDataFrame object as soon as this
        object is garbage collected. This helps avoiding conflicts between
        the garbage collector, the cppyy memory regulator and the C++ object
        destructor.
        """
        if hasattr(self, "rdf_node"):
            del self.rdf_node

    @property
    def npartitions(self) -> Optional[int]:
        return self._npartitions

    @npartitions.setter
    def npartitions(self, value: int) -> None:
        """
        The number of partitions for this dataframe is updated only if the user
        did not initially specify one when creating the dataframe.
        """
        if not self._user_specified_npartitions:
            self._npartitions = value

    def _prune_graph(self):
        """
        Prunes nodes from the graph under certain conditions. A node is pruned
        if it has no children and the user has no references to it. The internal
        representation of the graph is traversed in such a way so that leaf
        nodes are checked before their parents.
        """
        logger.debug("Starting computational graph pruning")
        self.graph_nodes = deque(node for node in self.graph_nodes if not node.is_prunable())
        logger.debug("Ended computational graph pruning")

    def _get_action_nodes(self) -> List[Node]:
        """Generates a list of nodes in the graph that are actions."""
        # This function is called after distributed execution of the graph, no
        # need to repeat the pruning
        action_nodes: List[Node] = []
        for node in reversed(self.graph_nodes):
            _append_node_to_actions(node.operation, node, action_nodes)
        return action_nodes

    def _generate_graph_dict(self) -> Dict[int, Node]:
        """
        Generates a dictionary holding information about all nodes in the graph.
        It is then given to the distributed scheduler.
        """
        # Need to prune the graph before sending it for distributed execution
        self._prune_graph()
        # We need to store references of each node id separately from the node
        # objects themselves. In a distributed task, we need to know for any
        # node which node is its parent (in order to execute an RDF operation
        # on the right node). We can't serialize the reference to the parent
        # node that each node object has, since that would trigger recursive
        # serialization of the parent(s) of parent(s) nodes.
        return {node.node_id: node for node in reversed(self.graph_nodes)}

    @abstractmethod
    def _build_ranges(self) -> List[Ranges.DataRange]:
        pass

    @abstractmethod
    def _generate_rdf_creator(self) -> Callable[[Ranges.DataRange], TaskObjects]:
        pass

    @abstractmethod
    def _handle_returned_values(self, values: TaskResult) -> Iterable:
        pass

    def _execute_and_retrieve_results(self, mapper, local_nodes) -> TaskResult:
        if self.drawables_dict is not None:
            # Prepare a dictionary with additional information for live visualization
            drawables_info_dict = {
                # Key: node_id
                node.node_id: (
                    # Tuple containing:
                    # 1. Callback functions passed by the user
                    self.drawables_dict[node.node_id],
                    # 2. Index of the node in the local_nodes list
                    i,
                    # 3. Name of the operation associated with the node
                    node.operation.name
                )
                for i, node in enumerate(local_nodes)
                # Filter: Only include nodes requested by the user
                if node.node_id in self.drawables_dict
            }
            return self.backend.ProcessAndMergeLive(self._build_ranges(), mapper, distrdf_reducer, drawables_info_dict)
        else:
            return self.backend.ProcessAndMerge(self._build_ranges(), mapper, distrdf_reducer)

    def execute_graph(self) -> None:
        """
        Executes an RDataFrame computation graph on a distributed backend.

        The needed ingredients are:

        - A collection of logical ranges in which the dataset is split. Each
          range is going to be assigned to a distributed task.
        - A representation of the computation graph that the task needs to
          execute.
        - A way to generate an RDataFrame instance starting from the logical
          range of the task.
        - Optionally, some setup code to be run at the beginning of each task.

        These are used as inputs to a generic mapper function. Results from the
        various mappers are then reduced and the final results are retrieved in
        the local session. These are properly handled to perform extra checks,
        depending on the data source. Finally, the local user-facing nodes are
        filled with the values that were computed distributedly so that they
        can be accessed in the application like with local RDataFrame.
        """

        # Updates the number of partitions for this dataframe if the user did
        # not specify one initially. This is done each time the computations are
        # triggered, in case the user changed the resource configuration
        # between runs (e.g. changing the number of available cores).
        self.npartitions = self.backend.optimize_npartitions()

        self.exec_id = _graph_cache.ExecutionIdentifier(self.rdf_uuid, uuid.uuid4())


        computation_graph_callable = partial(ComputationGraphGenerator.trigger_computation_graph, self._generate_graph_dict())

        # Accumulate all code that needs to be declared in one string
        code_to_declare = "\n".join(self.backend.strings_to_declare.values())

        mapper = partial(distrdf_mapper,
                         build_rdf_from_range=self._generate_rdf_creator(),
                         computation_graph_callable=computation_graph_callable,
                         initialization_fn=self.backend.initialization,
                         code_to_declare=code_to_declare)

        # List of action nodes in the same order as values
        local_nodes = self._get_action_nodes()

        # Execute graph distributedly and return the aggregated results from all tasks
        # using the appropriate backend method based on whether or not live visualization is enabled
        try:
            returned_values = self._execute_and_retrieve_results(mapper, local_nodes)
        finally:
            # Cleanup the current execution artifacts from the caches on the workers
            self.backend.cleanup_cache(self.exec_id)

        # Perform any extra checks that may be needed according to the
        # type of the head node
        final_values = self._handle_returned_values(returned_values)
        
        # Set the value of every action node
        for node, value in zip(local_nodes, final_values):
            Utils.set_value_on_node(value, node, self.backend)


def get_headnode(backend: BaseBackend, npartitions: int, *args) -> HeadNode:
    """
    A factory for different kinds of head nodes of the RDataFrame computation
    graph, depending on the arguments to the RDataFrame constructor. Parses the
    arguments and compares them against the possible RDataFrame constructors.
    """

    # Early check that arguments are accepted by RDataFrame
    try:
        localdf = ROOT.RDataFrame(*args)
    except TypeError:
        raise TypeError(("The arguments provided are not accepted by any RDataFrame constructor. "
                         "See the RDataFrame documentation for the accepted constructor argument types."))

    if isinstance(args[0], ROOT.RDF.Experimental.RDatasetSpec):
        return RDatasetSpecHeadNode(backend, npartitions, localdf, *args)

    label = ROOT.Internal.RDF.GetDataSourceLabel(ROOT.RDF.AsRNode(localdf))
    if label == "TTreeDS":
        return TreeHeadNode(backend, npartitions, localdf, *args)
    elif label == "RNTupleDS":
        return RNTupleHeadNode(backend, npartitions, localdf, *args)
    elif label == "EmptyDS":
        return EmptySourceHeadNode(backend, npartitions, localdf, args[0])
    else:
        raise RuntimeError(
            (f"First argument {args[0]} of type {type(args[0])} is not "
             "recognised as a supported argument for distributed RDataFrame. "
             "Currently supported data sources are: TTree, RNTuple or an empty "
             "data source."))


class EmptySourceHeadNode(HeadNode):
    """
    The head node of a computation graph where the RDataFrame data source is
    empty and a number of sequential entries will be created at runtime. This
    head node is responsible for the following RDataFrame constructor::

        RDataFrame(ULong64_t numEntries)

    Attributes:
        nentries (int): The number of sequential entries the RDataFrame will create.

        npartitions (int): The number of partitions the dataset will be split in
            for distributed execution.
    """

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], localdf: ROOT.RDataFrame, nentries: int):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            nentries (int): The number of entries this RDataFrame will process.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super().__init__(backend, npartitions, localdf)

        self.nentries = nentries

    def _build_ranges(self) -> List[Ranges.DataRange]:
        """Build the ranges for this dataset."""
        # Empty datasets cannot be processed distributedly
        if not self.nentries:
            raise RuntimeError(
                ("Cannot build a distributed RDataFrame with zero entries. "
                 "Distributed computation will fail. "))
        # TODO: This shouldn't be triggered if entries == 1. The current minimum
        # amount of partitions is 2. We need a robust reducer that smartly
        # becomes no-op if npartitions == 1 to avoid this.
        if self.npartitions > self.nentries:
            # Restrict 'npartitions' if it's greater than 'nentries'
            msg = ("Number of partitions {0} is greater than number of entries {1} "
                   "in the dataframe. Using {1} partition(s)".format(self.npartitions, self.nentries))
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.npartitions = self.nentries
        return Ranges.get_balanced_ranges(self.nentries, self.npartitions, self.exec_id)

    def _generate_rdf_creator(self) -> Callable[[Ranges.DataRange], TaskObjects]:
        """
        Generates a function that is responsible for building an instance of
        RDataFrame on a distributed mapper for a given entry range. Specific for
        an empty data source.
        """

        nentries = self.nentries

        def build_rdf_from_range(current_range):
            """
            Builds an RDataFrame instance for a distributed mapper.
            """
            if current_range.exec_id not in _graph_cache._RDF_REGISTER:
                _graph_cache._RDF_REGISTER[current_range.exec_id] = ROOT.RDataFrame(nentries)

            ROOT.Internal.RDF.ChangeEmptyEntryRange(
                ROOT.RDF.AsRNode(_graph_cache._RDF_REGISTER[current_range.exec_id]),
                (current_range.start, current_range.end))

            return TaskObjects(_graph_cache._RDF_REGISTER[current_range.exec_id], None)

        return build_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable:
        """
        Handle values returned after distributed execution. No extra checks are
        needed in the empty source case.
        """
        return values.mergeables


class TreeHeadNode(HeadNode):
    """
    The head node of a computation graph where the RDataFrame data source is
    a TTree or a TChain. This head node is responsible for the following
    RDataFrame constructors::

        RDataFrame(std::string_view treeName, std::string_view filenameglob, const ColumnNames_t &defaultBranches = {})
        RDataFrame(std::string_view treename, const std::vector<std::string> &fileglobs, const ColumnNames_t &defaultBranches = {})
        RDataFrame(std::string_view treeName, TDirectory *dirPtr, const ColumnNames_t &defaultBranches = {})
        RDataFrame(TTree &tree, const ColumnNames_t &defaultBranches = {})

    Attributes:
        npartitions (int): The number of partitions the dataset will be split in
            for distributed execution.

        tree (ROOT.TTree, ROOT.TChain): The dataset that will be processed. This
            is either stored as is if the user passed it as the first argument
            to the distributed RDataFrame constructor, or created from the
            arguments of the other constructor overloads.

        treename (str): The name of the dataset.

        inputfiles (list[str]): List of file names where the dataset is stored.

        defaultbranches (ROOT.std.vector[ROOT.std.string], None): Optional list
            of branches to be read from the tree passed by the user. Defaults to
            None.

        friendinfo (ROOT.Internal.TreeUtils.RFriendInfo, None): Optional
            information about friend trees of the dataset. Retrieved only if a
            TTree or TChain is passed to the constructor. Defaults to None.

    """

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], localdf: ROOT.RDataFrame, *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (iterable): Iterable with the arguments to the RDataFrame constructor.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super().__init__(backend, npartitions, localdf)

        self.defaultbranches = None
        # Information about friend trees, if they are present.
        self.friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo] = None

        # Retrieve the TTree/TChain that will be processed
        if isinstance(args[0], ROOT.TTree):
            # RDataFrame(tree, defaultBranches = {})
            tree = args[0]
            # TTreeIndex is not supported in distributed RDataFrame
            idx_found, friendname = ROOT.Internal.TreeUtils.TreeUsesIndexedFriends(
                tree)
            if idx_found:
                raise ValueError(
                    f"Friend tree '{friendname}' has a TTreeIndex. "
                    "This is not supported in distributed mode."
                )
            # Retrieve information about friend trees when user passes a TTree
            # or TChain object.
            fi = ROOT.Internal.TreeUtils.GetFriendInfo(args[0])
            self.friendinfo = fi if not fi.fFriendNames.empty() else None
            if len(args) == 2:
                self.defaultbranches = deepcopy(args[1])
        else:
            if isinstance(args[1], ROOT.TDirectory):
                # RDataFrame(treeName, dirPtr, defaultBranches = {})
                # We can assume both the argument TDirectory* and the TTree*
                # returned from TDirectory::Get are not nullptr since we already
                # did and early check of the user arguments in get_headnode
                tree = args[1].Get(args[0])
            elif isinstance(args[1], (str, ROOT.std.string_view)):
                # RDataFrame(treeName, filenameglob, defaultBranches = {})
                tree = ROOT.TChain(args[0])
                tree.Add(str(args[1]))
            elif isinstance(args[1], (list, ROOT.std.vector[ROOT.std.string])):
                # RDataFrame(treename, fileglobs, defaultBranches = {})
                tree = ROOT.TChain(args[0])
                for filename in args[1]:
                    tree.Add(str(filename))
            # In any of the three constructors considered in this branch, if
            # the user supplied three arguments then the third argument is a
            # list of default branches
            if len(args) == 3:
                self.defaultbranches = deepcopy(args[2])

        # maintreename: name of the tree or main name of the chain
        self.maintreename = str(tree.GetName())
        # subtreenames: names of all subtrees in the chain or full path to the tree in the file it belongs to
        self.subtreenames = [str(treename) for treename in ROOT.Internal.TreeUtils.GetTreeFullPaths(tree)]
        self.inputfiles = [str(filename) for filename in ROOT.Internal.TreeUtils.GetFileNamesFromTree(tree)]

    def _build_ranges(self) -> List[Ranges.DataRange]:
        """Build the ranges for this dataset."""
        logger.debug("Building ranges from dataset info:\n"
                     "main treename: %s\n"
                     "names of subtrees: %s\n"
                     "input files: %s\n", self.maintreename, self.subtreenames, self.inputfiles)

        if logger.isEnabledFor(logging.DEBUG):
            # Compute clusters and entries of the first tree in the dataset.
            # This will call once TFile::Open, but we pay this cost to get an estimate
            # on whether the number of requested partitions is reasonable.
            # Depending on the cluster setup, this may still be quite costly, so
            # we decide to pay the price only if the user explicitly requested
            # warning logging.
            clusters, entries = ROOT.Internal.TreeUtils.GetClustersAndEntries(self.subtreenames[0], self.inputfiles[0])
            # The file could contain an empty tree. In that case, the estimate will not be computed.
            if entries > 0:
                partitionsperfile = self.npartitions / len(self.inputfiles)
                if partitionsperfile > len(clusters):
                    logger.debug(
                        "The number of requested partitions could be higher than the maximum amount of "
                        "chunks the dataset can be split in. Some tasks could be doing no work. Consider "
                        "setting the 'npartitions' parameter of the RDataFrame constructor to a lower value.")

        return Ranges.get_percentage_ranges(self.subtreenames, self.inputfiles, self.npartitions, self.friendinfo, self.exec_id, None)

    def _generate_rdf_creator(self) -> Callable[[Ranges.DataRange], TaskObjects]:
        """
        Generates a function that is responsible for building an instance of
        RDataFrame on a distributed mapper for a given entry range. Specific for
        the TTree data source.
        """

        def attach_friend_info_if_present(current_range: Ranges.TreeRange,
                                          ds: ROOT.RDF.Experimental.RDatasetSpec) -> None:
            """
            Adds info about friend trees to the input chain. Also aligns the
            starting and ending entry of the friend chain cache to those of the
            main chain.
            """
            # Gather information about friend trees. Check that we got an
            # RFriendInfo struct and that it's not empty
            if (current_range.friendinfo is not None):
                # If the friend is a TChain, the zipped information looks like:
                # (name, alias), (file1.root, file2.root, ...), (subname1, subname2, ...)
                # If the friend is a TTree, the file list is made of
                # only one filename and the list of names of the sub trees
                # is empty, so the zipped information looks like:
                # (name, alias), (filename.root, ), ()
                zipped_friendinfo = zip(
                    current_range.friendinfo.fFriendNames,
                    current_range.friendinfo.fFriendFileNames,
                    current_range.friendinfo.fFriendChainSubNames
                )
                for (friend_name, friend_alias), friend_filenames, friend_chainsubnames in zipped_friendinfo:
                    friend_chainsubnames = friend_chainsubnames if len(friend_chainsubnames) > 0 else [friend_name]*len(friend_filenames)
                    ds.WithGlobalFriends(friend_chainsubnames, friend_filenames, friend_alias)

        def build_rdf_from_range(current_range: Ranges.TreeRangePerc) -> TaskObjects:
            """
            Builds an RDataFrame instance for a distributed mapper.

            The function creates a TChain from the information contained in the
            input range object. If the chain cannot be built, returns None.
            """

            clustered_range, entries_in_trees = Ranges.get_clustered_range_from_percs(current_range)

            if clustered_range is None:
                return TaskObjects(None, entries_in_trees)

            ds = ROOT.RDF.Experimental.RDatasetSpec()
            # add a sample with no name to represent the whole dataset
            ds.AddSample(("", clustered_range.treenames, clustered_range.filenames))
            ds.WithGlobalRange((clustered_range.globalstart, clustered_range.globalend))

            attach_friend_info_if_present(clustered_range, ds)

            if current_range.exec_id not in _graph_cache._RDF_REGISTER:
                # Fill the cache with the new RDataFrame
                _graph_cache._RDF_REGISTER[current_range.exec_id] = ROOT.RDataFrame(ds)
            else:
                # Update it to the range of entries for this task
                ROOT.Internal.RDF.ChangeSpec(
                    ROOT.RDF.AsRNode(_graph_cache._RDF_REGISTER[current_range.exec_id]),
                    ROOT.std.move(ds))

            return TaskObjects(_graph_cache._RDF_REGISTER[current_range.exec_id], entries_in_trees)

        return build_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable:
        """
        Handle values returned after distributed execution. When the data source
        is a TTree, check that exactly the input files and all the entries in
        the dataset were processed during distributed execution.
        """
        if values.mergeables is None:
            raise RuntimeError("The distributed execution returned no values. "
                               "This can happen if all files in your dataset contain empty trees.")

        # User could have requested to read the same file multiple times indeed
        input_files_and_trees = [
            f"{filename}/{treename}" for filename, treename in zip(self.inputfiles, self.subtreenames)
        ]
        files_counts = Counter(input_files_and_trees)

        entries_in_trees = values.entries_in_trees
        # Keys should be exactly the same
        if files_counts.keys() != entries_in_trees.trees_with_entries.keys():
            raise RuntimeError("The specified input files and the files that were "
                                "actually processed are not the same:\n"
                                f"Input files: {list(files_counts.keys())}\n"
                                f"Processed files: {list(entries_in_trees.trees_with_entries.keys())}")

        # Multiply the entries of each tree by the number of times it was
        # requested by the user
        for fullpath in files_counts:
            entries_in_trees.trees_with_entries[fullpath] *= files_counts[fullpath]

        total_dataset_entries = sum(entries_in_trees.trees_with_entries.values())
        if entries_in_trees.processed_entries != total_dataset_entries:
            raise RuntimeError(f"The dataset has {total_dataset_entries} entries, "
                               f"but {entries_in_trees.processed_entries} were processed.")

        return values.mergeables
    
class RDatasetSpecHeadNode(HeadNode):
    """
    The head node of a computation graph where the RDataFrame data source is
    an RDatasetSpec object. This head node is responsible for the following
    RDataFrame constructors:
        RDataFrame(ROOT.RDF.Experimental.RDatasetSpec spec)

    Attributes:
        npartitions (int): The number of partitions the dataset will be split in
            for distributed execution.

        rdatasetspec (ROOT.RDF.Experimental.RDatasetSpec): rdataset spec object
            used to construct the RDataFrame

        subtreenames (list[str]): List of tree names in the dataset.
        
        inputfiles (list[str]): List of file names where the dataset is stored.

        friendinfo (ROOT.Internal.TreeUtils.RFriendInfo, None): Optional
            information about friend trees of the dataset. Retrieved from 
            RDatasetSpec.GetFriendInfo(). Defaults to None.
    """

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], localdf: ROOT.RDataFrame, *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (iterable): Iterable with the arguments to the RDataFrame constructor.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super().__init__(backend, npartitions, localdf)

        # Information about friend trees, if they are present.
        self.sampleMap: Dict[str, Ranges.SerializableRSample] = None 
        
        self.friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo] = None
                
            
        # Retrieve the RDatasetSpec that will be processed
        if isinstance(args[0], ROOT.RDF.Experimental.RDatasetSpec):
            # RDataFrame(rdatasetspec)
            fi = args[0].GetFriendInfo()
            self.friendinfo = fi if not fi.fFriendNames.empty() else None

        else:
            raise RuntimeError(
                f"First argument {args[0]} of type {type(args[0])} is not supported "
                "in RDatasetSpecHeaNode")
                
        if (args[0].GetEntryRangeBegin() != 0):
            warnings.warn("Setting global range is not supported in distributed parsing of RDatasetSpec, the set range will be ignored and all events will be processed.")
            
        # subtreenames: names of all subtrees in the chain or full path to the tree in the file it belongs to
        self.subtreenames = [str(treename) for treename in args[0].GetTreeNames()]
        self.inputfiles = [str(filename) for filename in args[0].GetFileNameGlobs()]
            
        # A map to map file id (filename/tree) to SerializableRSamples
        self.sampleMap = {}
        samples = ROOT.Internal.RDF.MoveOutSamples(args[0])

        for sample in samples:
            trees = sample.GetTreeNames()
            files = sample.GetFileNameGlobs()
            for file, tree in zip(files, trees): 
                sampleId = file + "/" + tree
                self.sampleMap.update({sampleId : Ranges.SerializableRSample(sample)})

    def _build_ranges(self) -> List[Ranges.DataRange]:
        """Build the ranges for this dataset."""
        logger.debug("Building ranges from dataset info:\n"
                     "names of subtrees: %s\n"
                     "input files: %s\n", self.subtreenames, self.inputfiles)

        if logger.isEnabledFor(logging.DEBUG):
            # Compute clusters and entries of the first tree in the dataset.
            # This will call once TFile::Open, but we pay this cost to get an estimate
            # on whether the number of requested partitions is reasonable.
            # Depending on the cluster setup, this may still be quite costly, so
            # we decide to pay the price only if the user explicitly requested
            # warning logging.
            clusters, entries = ROOT.Internal.TreeUtils.GetClustersAndEntries(
                self.subtreenames[0], self.inputfiles[0])
            # The file could contain an empty tree. In that case, the estimate will not be computed.
            if entries > 0:
                partitionsperfile = self.npartitions / len(self.inputfiles)
                if partitionsperfile > len(clusters):
                    logger.debug(
                        "The number of requested partitions could be higher than the maximum amount of "
                        "chunks the dataset can be split in. Some tasks could be doing no work. Consider "
                        "setting the 'npartitions' parameter of the RDataFrame constructor to a lower value.")
        
        return Ranges.get_percentage_ranges(self.subtreenames, self.inputfiles, self.npartitions, self.friendinfo, self.exec_id, self.sampleMap)

    def _generate_rdf_creator(self) -> Callable[[Ranges.DataRange], TaskObjects]:
        """
        Generates a function that is responsible for building an instance of
        RDataFrame on a distributed mapper for a given entry range.
        """

        def attach_friend_info_if_present(current_range: Ranges.TreeRange,
                                          ds: ROOT.RDF.Experimental.RDatasetSpec) -> None:
            """
            Adds info about friend trees to the input chain. Also aligns the
            starting and ending entry of the friend chain cache to those of the
            main chain.
            """
            # Gather information about friend trees. Check that we got an
            # RFriendInfo struct and that it's not empty
            if (current_range.friendinfo is not None):
                # If the friend is a TChain, the zipped information looks like:
                # (name, alias), (file1.root, file2.root, ...), (subname1, subname2, ...)
                # If the friend is a TTree, the file list is made of
                # only one filename and the list of names of the sub trees
                # is empty, so the zipped information looks like:
                # (name, alias), (filename.root, ), ()
                zipped_friendinfo = zip(
                    current_range.friendinfo.fFriendNames,
                    current_range.friendinfo.fFriendFileNames,
                    current_range.friendinfo.fFriendChainSubNames
                )
                for (friend_name, friend_alias), friend_filenames, friend_chainsubnames in zipped_friendinfo:
                    friend_chainsubnames = (
                        friend_chainsubnames if len(friend_chainsubnames) > 0
                        else [friend_name]*len(friend_filenames)
                    )
                    ds.WithGlobalFriends(
                        friend_chainsubnames, friend_filenames, friend_alias)
                    
        def build_rdf_from_range(current_range: Ranges.TreeRangePerc) -> TaskObjects:
            """
            Builds an RDataFrame instance for a distributed mapper.

            The function creates a TChain from the information contained in the
            input range object. If the chain cannot be built, returns None.
            """

            clustered_range, entries_in_trees = Ranges.get_clustered_range_from_percs(
                current_range)

            if clustered_range is None:
                return TaskObjects(None, entries_in_trees)
            
            ds = ROOT.RDF.Experimental.RDatasetSpec()
        
            filenames_cluster = clustered_range.filenames
            treenames_cluster = clustered_range.treenames

            # The goal is to correctly assign the metadata information to the event being
            # processed.
            # The distributed tasks are split based on entries in the processed files, we get the task ranges.
            # Files, however, also belong to various samples. We need to make sure the files are always
            # correctly matched with the samples they are coming from as this is essential for the correct metadata processing.
            # In order to do that, we need to find the common files (good_files) from the list of
            # files processed in a given task range (filenames_cluster) and the list of files belonging to the given sample
            # (mysample.GetFilenameGlobs). Additionally, we need a dictionary to correctly match files with their trees.

            # Matching files with trees 
            treesInFileMap = {}
            fileids_cluster = []
        
            for filename, treename in zip(filenames_cluster, treenames_cluster):
                fileid = filename + "/" + treename
                fileids_cluster.append(fileid)
                treesInFileMap.update({fileid : treename})

            # Making sure we don't double count samples  
            unique_samples = [] 
            samplenames = []
            
            for sample in clustered_range.samples:
                mysample = sample._sample
                samplename = mysample.GetSampleName()
                if samplename not in samplenames:
                    unique_samples.append(mysample)
                    samplenames.append(samplename)
            
            # Core: matching files with samples and adding samples to the spec that will be processed
            for mysample in unique_samples:
                sample_fileids = [
                    filenameglob + "/" + treename
                    for filenameglob, treename in zip(mysample.GetFileNameGlobs(), mysample.GetTreeNames())
                ]
                    
                good_files = list((Counter(filenames_cluster) & Counter(mysample.GetFileNameGlobs())).elements())
                good_fileids = list((Counter(fileids_cluster) & Counter(sample_fileids)).elements()) 

                good_trees =[
                    treesInFileMap.get(fileid) for fileid in good_fileids
                ]
                
                ds.AddSample((mysample.GetSampleName(), good_trees, good_files, mysample.GetMetaData()))
                
            ds.WithGlobalRange(
                (clustered_range.globalstart, clustered_range.globalend))
            
            attach_friend_info_if_present(clustered_range, ds)

            if current_range.exec_id not in _graph_cache._RDF_REGISTER:
                # Fill the cache with the new RDataFrame
                _graph_cache._RDF_REGISTER[current_range.exec_id] = ROOT.RDataFrame(ds) 
                
            else:
                # Update it to the range of entries for this task
                ROOT.Internal.RDF.ChangeSpec(
                    ROOT.RDF.AsRNode( _graph_cache._RDF_REGISTER[current_range.exec_id]),
                    ROOT.std.move(ds))

            return TaskObjects(_graph_cache._RDF_REGISTER[current_range.exec_id], entries_in_trees)

        return build_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable:
        """
        Handle values returned after distributed execution. When the data source
        is a TTree, check that exactly the input files and all the entries in
        the dataset were processed during distributed execution.
        """
        if values.mergeables is None:
            raise RuntimeError("The distributed execution returned no values. "
                               "This can happen if all files in your dataset contain empty trees.")

        # User could have requested to read the same file multiple times indeed
        input_files_and_trees = [
            f"{filename}/{treename}" for filename, treename in zip(self.inputfiles, self.subtreenames)
        ]
        files_counts = Counter(input_files_and_trees)

        entries_in_trees = values.entries_in_trees
        # Keys should be exactly the same
        if files_counts.keys() != entries_in_trees.trees_with_entries.keys():
            raise RuntimeError("The specified input files and the files that were "
                               "actually processed are not the same:\n"
                               f"Input files: {list(files_counts.keys())}\n"
                               f"Processed files: {list(entries_in_trees.trees_with_entries.keys())}")

        # Multiply the entries of each tree by the number of times it was
        # requested by the user
        for fullpath in files_counts:
            entries_in_trees.trees_with_entries[fullpath] *= files_counts[fullpath]

        total_dataset_entries = sum(
            entries_in_trees.trees_with_entries.values())
        if entries_in_trees.processed_entries != total_dataset_entries:
            raise RuntimeError(f"The dataset has {total_dataset_entries} entries, "
                               f"but {entries_in_trees.processed_entries} were processed.")

        return values.mergeables
    
class RNTupleHeadNode(HeadNode):
    """
    The head node of a computation graph where the RDataFrame data source is
    an RNtuple.
    """

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], localdf: ROOT.RDataFrame, *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (iterable): Iterable with the arguments to the RDataFrame constructor.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super().__init__(backend, npartitions, localdf)

        self.mainntuplename = args[0]
        self.inputfiles = args[1]
        # Keep this in accordance with the implementation of TTree for now
        self.subnames = [self.mainntuplename] * len(self.inputfiles)

    def _build_ranges(self) -> List[Ranges.DataRange]:
        # """Build the ranges for this dataset."""
        # For the moment, we explicitly pass only one "subname", since there is
        # only one name possible for the whole RNTuple
        # TODO: implement action cloning for RNTuple
        # Inject a sentinel flag in the execution identifier to specify this
        # RDataFrame instance is going to process RNTuple data and the computation
        # graph needs to be recreated at every task
        self.exec_id = _graph_cache.ExecutionIdentifier("RNTuple", self.exec_id.graph_uuid)
        return Ranges.get_ntuple_ranges(self.mainntuplename, self.inputfiles, self.npartitions, self.exec_id)
    
    def _generate_rdf_creator(self) -> Callable[[Ranges.DataRange], TaskObjects]:
        """
        Generates a function that is responsible for building an instance of
        RDataFrame on a distributed mapper for a given entry range. Specific for
        the RNtuple data source.
        """

        def build_rdf_from_range(current_range: Ranges.get_ntuple_ranges) -> TaskObjects:
            """
            Builds an RDataFrame instance for a distributed mapper.

            The function creates a TChain from the information contained in the
            input range object. If the chain cannot be built, returns None.
            """
            ntuplename, filenames = current_range.ntuplename, current_range.filenames            
            if not filenames:
                return TaskObjects(None, None)

            return TaskObjects(ROOT.RDF.FromRNTuple(ntuplename, filenames), None)

        return build_rdf_from_range

    def _handle_returned_values(self, values: TaskResult) -> Iterable:
        """
        Handle values returned after distributed execution. When the data source
        is an RNtuple, check that exactly the input files and all the entries in
        the dataset were processed during distributed execution.
        """
        if values.mergeables is None:
            raise RuntimeError("The distributed execution returned no values. "
                               "This can happen if all files in your dataset contain empty trees.")

        return values.mergeables        
