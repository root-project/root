import logging
import warnings

from itertools import zip_longest
from typing import Callable, List, Optional

import ROOT

from DistRDF import Ranges
from DistRDF.Node import HeadNode

logger = logging.getLogger(__name__)


def get_headnode(npartitions: int, *args) -> HeadNode:
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
        return EmptySourceHeadNode(npartitions, firstarg)
    elif isinstance(firstarg, (ROOT.TTree, str)):
        # RDataFrame(std::string_view treeName, filenameglob, defaultBranches = {})
        # RDataFrame(std::string_view treename, filenames, defaultBranches = {})
        # RDataFrame(std::string_view treeName, dirPtr, defaultBranches = {})
        # RDataFrame(TTree &tree, const ColumnNames_t &defaultBranches = {})
        return TreeHeadNode(npartitions, *args)
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

    def __init__(self, npartitions, nentries):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            nentries (int): The number of entries this RDataFrame will process.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super(EmptySourceHeadNode, self).__init__(None, None)

        self.nentries = nentries
        self.npartitions = npartitions

    def build_ranges(self):
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

    def generate_rdf_creator(self):
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
            return ROOT.RDataFrame(nentries).Range(current_range.start, current_range.end)

        return build_rdf_from_range


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

    def __init__(self, npartitions, *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (iterable): Iterable with the arguments to the RDataFrame constructor.

            npartitions (int): The number of partitions the dataset will be
                split in for distributed execution.
        """
        super(TreeHeadNode, self).__init__(None, None)

        self.npartitions = npartitions

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

    def build_ranges(self) -> List[Ranges.TreeRangePerc]:
        """Build the ranges for this dataset."""
        logger.debug("Building ranges from dataset info:\n"
                     "main treename: %s\n"
                     "names of subtrees: %s\n"
                     "input files: %s\n", self.maintreename, self.subtreenames, self.inputfiles)

        if logger.isEnabledFor(logging.WARNING):
            # Compute clusters and entries of the first tree in the dataset.
            # This will call once TFile::Open, but we pay this cost to get an estimate
            # on whether the number of requested partitions is reasonable.
            # Depending on the cluster setup, this may still be quite costly.
            clusters, entries = Ranges.get_clusters_and_entries(self.subtreenames[0], self.inputfiles[0])
            if entries is not None:
                # The file could contain an empty tree. In that case, the estimate will not be computed.
                partitionsperfile = self.npartitions / len(self.inputfiles)
                if partitionsperfile > len(clusters):
                    logger.warning(
                        "The number of requested partitions could be higher than the maximum amount of "
                        "chunks the dataset can be split in. Some tasks could be doing no work. Consider "
                        "setting the 'npartitions' parameter of the RDataFrame constructor to a lower value.")

        return Ranges.get_percentage_ranges(self.subtreenames, self.inputfiles, self.npartitions, self.friendinfo)

    def generate_rdf_creator(self) -> Callable[[Ranges.TreeRangePerc], Optional[ROOT.RDataFrame]]:
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

        def build_chain_from_range(current_range: Ranges.TreeRangePerc) -> Optional[ROOT.TChain]:
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

            clustered_range = Ranges.get_clustered_range_from_percs(current_range)
            if clustered_range is None:
                # The task could not be correctly built, don't create the TChain
                return None

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

            return chain

        def build_rdf_from_range(current_range: Ranges.TreeRangePerc) -> Optional[ROOT.RDataFrame]:
            """
            Builds an RDataFrame instance for a distributed mapper.

            The function creates a TChain from the information contained in the
            input range object. If the chain cannot be built, returns None.
            """
            # Prepare main TChain
            chain = build_chain_from_range(current_range)
            if chain is None:
                # The chain could not be built.
                return None

            # Create RDataFrame object for this task
            if defaultbranches is not None:
                rdf = ROOT.RDataFrame(chain, defaultbranches)
            else:
                rdf = ROOT.RDataFrame(chain)

            # Bind the TChain to the RDataFrame object before returning it. Not
            # doing so would lead to the TChain being destroyed when leaving
            # the scope of this function.
            rdf._chain_lifeline = chain

            return rdf

        return build_rdf_from_range
