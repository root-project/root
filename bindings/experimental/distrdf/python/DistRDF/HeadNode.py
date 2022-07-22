from abc import ABC, abstractmethod
import logging
import warnings

from collections import Counter, deque
from dataclasses import dataclass
from functools import partial, singledispatch
from itertools import zip_longest
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

import ROOT

from DistRDF import ComputationGraphGenerator, Ranges
from DistRDF.Backends.Base import BaseBackend, distrdf_mapper, distrdf_reducer, TaskResult
from DistRDF.Node import Node
from DistRDF.Operation import Action, InstantAction, Operation
from DistRDF.Backends import Utils

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

    def __init__(self, backend: BaseBackend, npartitions: Optional[int]):
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

        # Internal attribute to keep track of the number of partitions. We also
        # check whether it was specified by the user when creating the dataframe.
        # If so, this attribute will not be updated when triggering.
        self._npartitions = npartitions
        self._user_specified_npartitions = True if npartitions is not None else False

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
    def _handle_returned_values(self, values: "TaskResult") -> Iterable:
        pass

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
        # Check if the workflow must be generated in optimized mode
        optimized = ROOT.RDF.Experimental.Distributed.optimized

        # Updates the number of partitions for this dataframe if the user did
        # not specify one initially. This is done each time the computations are
        # triggered, in case the user changed the resource configuration
        # between runs (e.g. changing the number of available cores).
        self.npartitions = self.backend.optimize_npartitions()

        if optimized:
            computation_graph_callable = partial(
                ComputationGraphGenerator.run_with_cppworkflow, self._generate_graph_dict())
        else:
            computation_graph_callable = partial(
                ComputationGraphGenerator.trigger_computation_graph, self._generate_graph_dict())

        mapper = partial(distrdf_mapper,
                         build_rdf_from_range=self._generate_rdf_creator(),
                         computation_graph_callable=computation_graph_callable,
                         initialization_fn=self.backend.initialization,
                         optimized=optimized)

        # Execute graph distributedly and return the aggregated results from all
        # tasks
        returned_values = self.backend.ProcessAndMerge(self._build_ranges(), mapper, distrdf_reducer)
        # Perform any extra checks that may be needed according to the
        # type of the head node
        final_values = self._handle_returned_values(returned_values)
        # List of action nodes in the same order as values
        local_nodes = self._get_action_nodes()
        # Set the value of every action node
        for node, value in zip(local_nodes, final_values):
            Utils.set_value_on_node(value, node, self.backend)


def get_headnode(backend: BaseBackend, npartitions: int, *args) -> HeadNode:
    """
    A factory for different kinds of head nodes of the RDataFrame computation
    graph, depending on the arguments to the RDataFrame constructor. Currently
    can return a TreeHeadNode or an EmptySourceHeadNode. Parses the arguments and
    compares them against the possible RDataFrame constructors.
    """

    # Early check that arguments are accepted by RDataFrame
    try:
        ROOT.RDataFrame(*args)
    except TypeError:
        raise TypeError(("The arguments provided are not accepted by any RDataFrame constructor. "
                         "See the RDataFrame documentation for the accepted constructor argument types."))

    firstarg = args[0]
    if isinstance(firstarg, int):
        # RDataFrame(ULong64_t numEntries)
        return EmptySourceHeadNode(backend, npartitions, firstarg)
    elif isinstance(firstarg, (ROOT.TTree, str)):
        # RDataFrame(std::string_view treeName, filenameglob, defaultBranches = {})
        # RDataFrame(std::string_view treename, filenames, defaultBranches = {})
        # RDataFrame(std::string_view treeName, dirPtr, defaultBranches = {})
        # RDataFrame(TTree &tree, const ColumnNames_t &defaultBranches = {})
        return TreeHeadNode(backend, npartitions, *args)
    else:
        raise RuntimeError(
            ("First argument {} of type {} is not recognised as a supported "
                "argument for distributed RDataFrame. Currently only TTree/Tchain "
                "based datasets or datasets created from a number of entries "
                "can be processed distributedly.").format(firstarg, type(firstarg)))


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

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], nentries: int):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            nentries (int): The number of entries this RDataFrame will process.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super().__init__(backend, npartitions)

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
        return Ranges.get_balanced_ranges(self.nentries, self.npartitions)

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
            return TaskObjects(ROOT.RDataFrame(nentries).Range(current_range.start, current_range.end), None)

        return build_rdf_from_range

    def _handle_returned_values(self, values: "TaskResult") -> Iterable:
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

    def __init__(self, backend: BaseBackend, npartitions: Optional[int], *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (iterable): Iterable with the arguments to the RDataFrame constructor.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super().__init__(backend, npartitions)

        self.defaultbranches = None
        self.friendinfo = None

        # Retrieve the TTree/TChain that will be processed
        if isinstance(args[0], ROOT.TTree):
            # RDataFrame(tree, defaultBranches = {})
            self.tree = args[0]
            # Retrieve information about friend trees when user passes a TTree
            # or TChain object.
            self.friendinfo = ROOT.Internal.TreeUtils.GetFriendInfo(args[0])
            if len(args) == 2:
                self.defaultbranches = args[1]
        else:
            if isinstance(args[1], ROOT.TDirectory):
                # RDataFrame(treeName, dirPtr, defaultBranches = {})
                # We can assume both the argument TDirectory* and the TTree*
                # returned from TDirectory::Get are not nullptr since we already
                # did and early check of the user arguments in get_headnode
                self.tree = args[1].Get(args[0])
            elif isinstance(args[1], (str, ROOT.std.string_view)):
                # RDataFrame(treeName, filenameglob, defaultBranches = {})
                self.tree = ROOT.TChain(args[0])
                self.tree.Add(str(args[1]))
            elif isinstance(args[1], (list, ROOT.std.vector[ROOT.std.string])):
                # RDataFrame(treename, fileglobs, defaultBranches = {})
                self.tree = ROOT.TChain(args[0])
                for filename in args[1]:
                    self.tree.Add(str(filename))
            # In any of the three constructors considered in this branch, if
            # the user supplied three arguments then the third argument is a
            # list of default branches
            if len(args) == 3:
                self.defaultbranches = args[2]

        # maintreename: name of the tree or main name of the chain
        self.maintreename = self.tree.GetName()
        # subtreenames: names of all subtrees in the chain or full path to the tree in the file it belongs to
        self.subtreenames = [str(treename) for treename in ROOT.Internal.TreeUtils.GetTreeFullPaths(self.tree)]
        self.inputfiles = [str(filename) for filename in ROOT.Internal.TreeUtils.GetFileNamesFromTree(self.tree)]

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
            clusters, entries = Ranges.get_clusters_and_entries(self.subtreenames[0], self.inputfiles[0])
            # The file could contain an empty tree. In that case, the estimate will not be computed.
            if entries > 0:
                partitionsperfile = self.npartitions / len(self.inputfiles)
                if partitionsperfile > len(clusters):
                    logger.debug(
                        "The number of requested partitions could be higher than the maximum amount of "
                        "chunks the dataset can be split in. Some tasks could be doing no work. Consider "
                        "setting the 'npartitions' parameter of the RDataFrame constructor to a lower value.")

        return Ranges.get_percentage_ranges(self.subtreenames, self.inputfiles, self.npartitions, self.friendinfo)

    def _generate_rdf_creator(self) -> Callable[[Ranges.DataRange], TaskObjects]:
        """
        Generates a function that is responsible for building an instance of
        RDataFrame on a distributed mapper for a given entry range. Specific for
        the TTree data source.
        """
        maintreename = self.maintreename
        defaultbranches = self.defaultbranches

        def attach_friend_info_if_present(current_range: Ranges.TreeRange, chain: ROOT.TChain) -> None:
            """
            Adds info about friend trees to the input chain. Also aligns the
            starting and ending entry of the friend chain cache to those of the
            main chain.
            """
            # Gather information about friend trees. Check that we got an
            # RFriendInfo struct and that it's not empty
            if (current_range.friendinfo is not None and
                    not current_range.friendinfo.fFriendNames.empty()):
                # Zip together the information about friend trees. Each
                # element of the iterator represents a single friend tree.
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
                    # Start a TChain with the current friend treename
                    friend_chain = ROOT.TChain(str(friend_name))
                    # Add each corresponding file to the TChain
                    # Use zip_longest to address both cases:
                    # - Friend is a TTree, filenames is a vector of length one
                    #   and chainsubnames is an empty vector.
                    # - Friend is a TChain, filenames and chainsubnames are
                    #   vectors of the same length.
                    for filename, chainsubname in zip_longest(friend_filenames, friend_chainsubnames, fillvalue=""):
                        fullpath = filename + "?#" + chainsubname
                        friend_chain.Add(str(fullpath))

                    # Set cache on the same range as the parent TChain
                    friend_chain.SetCacheEntryRange(current_range.globalstart, current_range.globalend)
                    # Finally add friend TChain to the parent (with alias)
                    chain.AddFriend(friend_chain, friend_alias)

        def build_chain_from_range(
                current_range: Ranges.TreeRangePerc) -> Tuple[Optional[ROOT.TChain], Ranges.TaskTreeEntries]:
            """
            Builds a TChain from the information in 'current_range'.

            Processing on the chain is restricted to the entries selected for
            this task via TEntryList. If the user provided info about friend
            trees, also that is used to attach the friends to the main chain.
            """

            # Build TEntryList for this range:
            elists = ROOT.TEntryList()

            # Build TChain of files for this range:
            chain = ROOT.TChain(maintreename)

            clustered_range, entries_in_trees = Ranges.get_clustered_range_from_percs(current_range)
            if clustered_range is None:
                # The task could not be correctly built, don't create the TChain
                return None, entries_in_trees

            for subtreename, filename, treenentries, start, end in zip(
                    clustered_range.treenames, clustered_range.filenames, clustered_range.treesnentries,
                    clustered_range.localstarts, clustered_range.localends):

                # Use default constructor of TEntryList rather than the
                # constructor accepting treename and filename, otherwise
                # the TEntryList would remove any url or protocol from the
                # file name.
                elist = ROOT.TEntryList()
                elist.SetTreeName(subtreename)
                elist.SetFileName(filename)
                elist.EnterRange(start, end)
                elists.AddSubList(elist)
                chain.Add(filename + "?#" + subtreename, treenentries)

            # We assume 'end' is exclusive
            chain.SetCacheEntryRange(clustered_range.globalstart, clustered_range.globalend)

            # Connect the entry list to the chain
            chain.SetEntryList(elists, "sync")

            # Needs the same globalstart and globalend of the chain created in
            # this task
            attach_friend_info_if_present(clustered_range, chain)

            return chain, entries_in_trees

        def build_rdf_from_range(current_range: Ranges.TreeRangePerc) -> TaskObjects:
            """
            Builds an RDataFrame instance for a distributed mapper.

            The function creates a TChain from the information contained in the
            input range object. If the chain cannot be built, returns None.
            """
            # Prepare main TChain
            chain, entries_in_trees = build_chain_from_range(current_range)
            if chain is None:
                # The chain could not be built.
                return TaskObjects(None, entries_in_trees)

            # Create RDataFrame object for this task
            if defaultbranches is not None:
                rdf = ROOT.RDataFrame(chain, defaultbranches)
            else:
                rdf = ROOT.RDataFrame(chain)

            # Bind the TChain to the RDataFrame object before returning it. Not
            # doing so would lead to the TChain being destroyed when leaving
            # the scope of this function.
            rdf._chain_lifeline = chain

            return TaskObjects(rdf, entries_in_trees)

        return build_rdf_from_range

    def _handle_returned_values(self, values: "TaskResult") -> Iterable:
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
            f"{filename}?#{treename}" for filename, treename in zip(self.inputfiles, self.subtreenames)
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
