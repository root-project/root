import glob
import logging
import warnings

import ROOT

from DistRDF import Node
from DistRDF import Ranges

logger = logging.getLogger(__name__)


class FriendInfo(object):
    """
    A simple class to hold information about friend trees.

    Attributes:
        friend_names (list): A list with the names of the `ROOT.TTree` objects
            which are friends of the main `ROOT.TTree`.

        friend_file_names (list): A list with the paths to the files
            corresponding to the trees in the `friend_names` attribute. Each
            element of `friend_names` can correspond to multiple file names.
    """

    def __init__(self, friend_names=[], friend_file_names=[]):
        """
        Create an instance of FriendInfo

        Args:
            friend_names (list): A list containing the treenames of the friend
                trees.

            friend_file_names (list): A list containing the file names
                corresponding to a given treename in friend_names. Each
                treename can correspond to multiple file names.
        """
        self.friend_names = friend_names
        self.friend_file_names = friend_file_names

    def __bool__(self):
        """
        Define the behaviour of FriendInfo instance when boolean evaluated.
        Both lists have to be non-empty in order to return True.

        Returns:
            bool: True if both lists are non-empty, False otherwise.
        """
        return bool(self.friend_names) and bool(self.friend_file_names)

    def __nonzero__(self):
        """
        Python 2 dunder method for __bool__. Kept for compatibility.
        """
        return self.__bool__()


class Factory(object):
    """
    A factory for different kinds of root nodes of the RDataFrame computation
    graph, depending on the arguments to the RDataFrame constructor. Currently
    can create a TreeHeadNode or an EntriesHeadNode.
    """

    @staticmethod
    def get_headnode(*args):
        """
        Parses the arguments and compares them against the possible RDataFrame
        constructors
        """

        # Early check that arguments are accepted by RDataFrame
        ROOT.RDataFrame(*args)

        firstarg = args[0]
        if isinstance(firstarg, int):
            # RDataFrame(ULong64_t numEntries)
            return EntriesHeadNode(firstarg)
        elif isinstance(firstarg, (ROOT.TTree, str)):
            # RDataFrame(std::string_view treeName, filenameglob, defaultBranches = {})
            # RDataFrame(std::string_view treename, filenames, defaultBranches = {})
            # RDataFrame(std::string_view treeName, dirPtr, defaultBranches = {})
            # RDataFrame(TTree &tree, const ColumnNames_t &defaultBranches = {})
            return TreeHeadNode(*args)
        else:
            raise RuntimeError(
                ("First argument {} of type {} is not recognised as a supported "
                 "argument for distributed RDataFrame. Currently only TTree/Tchain "
                 "based datasets or datasets created from a number of entries "
                 "can be processed distributedly.").format(firstarg, type(firstarg)))


class EntriesHeadNode(Node.Node):
    """
    The head node of a computation graph where the RDataFrame is created from
    a number of entries.
    """

    def __init__(self, nentries):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            nentries (int): The number of entries this RDataFrame will process.
        """
        super(EntriesHeadNode, self).__init__(None, None)

        self.nentries = nentries
        # Set at creation of the dataframe, might be optimized by the backend
        # in optimize_partitions
        self.npartitions = 2

    def build_ranges(self):
        """Build the ranges for this dataset."""
        # Empty datasets cannot be processed distributedly
        if not self.nentries:
            raise RuntimeError(
                ("Cannot build a distributed RDataFrame with zero entries. "
                 "Distributed computation will fail. "))
        if self.npartitions > self.nentries:
            # Restrict 'npartitions' if it's greater than 'nentries'
            msg = ("Number of partitions {0} is greater than number of entries {1} "
                   "in the dataframe. Using {1} partition(s)".format(self.npartitions, self.nentries))
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.npartitions = self.nentries
        return Ranges.get_balanced_ranges(self.nentries, self.npartitions)


class TreeHeadNode(Node.Node):
    """
    The Python equivalent of ROOT C++'s
    RDataFrame class.

    Attributes:
        args (list): A list of arguments that were provided to construct
            the RDataFrame object.


    DistRDF's RDataFrame constructor accepts the same arguments as the ROOT's
    RDataFrame constructor (see
    `RDataFrame <https://root.cern/doc/master/classROOT_1_1RDataFrame.html>`_)
    """

    def __init__(self, *args):
        """
        Creates a new RDataFrame instance for the given arguments.

        Args:
            *args (list): Variable length argument list to construct the
                RDataFrame object.
        """
        super(TreeHeadNode, self).__init__(None, None)

        # Keep the arguments to parse them when needed
        self.args = args
        # Set at creation of the dataframe, might be optimized by the backend
        # in optimize_partitions
        self.npartitions = 2

    @property
    def tree(self):
        """Tree instance if present"""
        return self.get_tree()

    @property
    def treename(self):
        """Name of the tree"""
        return self.get_treename()

    @property
    def nentries(self):
        """Entries of this dataset"""
        return self.get_num_entries()

    @property
    def defaultbranches(self):
        """Default branches selected by the user in the constructor"""
        return self.get_branches()

    @property
    def inputfiles(self):
        """List of input files of the dataset if any"""
        return self.get_inputfiles()

    def get_branches(self):
        """Gets list of default branches if passed by the user."""
        # ROOT Constructor:
        # RDataFrame(TTree& tree, defaultBranches = {})
        if len(self.args) == 2 and isinstance(self.args[0], ROOT.TTree):
            return self.args[1]
        # ROOT Constructors:
        # RDataFrame(treeName, filenameglob, defaultBranches = {})
        # RDataFrame(treename, filenames, defaultBranches = {})
        # RDataFrame(treeName, dirPtr, defaultBranches = {})
        if len(self.args) == 3:
            return self.args[2]

        return None

    def get_num_entries(self):
        """
        Gets the number of entries in the given dataset.

        Returns:
            int: This is the computed number of entries in the input dataset.

        """

        # First argument can be a string or a TTree/TChain
        firstarg = self.args[0]
        if isinstance(firstarg, ROOT.TTree):
            # If the argument is a TTree or TChain,
            # get the number of entries from it.
            return firstarg.GetEntries()
        elif isinstance(firstarg, str):
            # Construct a ROOT.TChain object with the name of the tree
            chain = ROOT.TChain(firstarg)

            # Add the list of input files to the chain
            for fname in self.inputfiles:
                chain.Add(fname)

            return chain.GetEntries()
        else:
            raise RuntimeError("Could not retrieve number of entries from the dataset.")

    def get_treename(self):
        """
        Get name of the TTree.

        Returns:
            (str, None): Name of the TTree, or :obj:`None` if there is no tree.

        """
        first_arg = self.args[0]
        if isinstance(first_arg, ROOT.TChain):
            # Get name from a given TChain
            return first_arg.GetName()
        elif isinstance(first_arg, ROOT.TTree):
            # Get name directly from the TTree
            return first_arg.GetUserInfo().At(0).GetName()
        elif isinstance(first_arg, str):
            # First argument was the name of the tree
            return first_arg
        else:
            raise RuntimeError("Could not find name of the input tree.")

    def get_tree(self):
        """
        Get ROOT.TTree instance used as an argument to DistRDF.RDataFrame()

        Returns:
            (ROOT.TTree, None): instance of the tree used to instantiate the
            RDataFrame, or `None` if another object was used. ROOT.Tchain
            inherits from ROOT.TTree so that can be the return value as well.
        """
        first_arg = self.args[0]
        if isinstance(first_arg, ROOT.TTree):
            return first_arg

        return None

    def get_inputfiles(self):
        """
        Get list of input files.

        This list can be extracted from a given TChain or from the list of
        arguments.

        Returns:
            (list, None): List of files, or None if there are no input files.

        """
        firstarg = self.args[0]
        if isinstance(firstarg, ROOT.TChain):
            # Extract file names from a given TChain
            chain = firstarg
            return [chainElem.GetTitle()
                    for chainElem in chain.GetListOfFiles()]
        elif isinstance(firstarg, ROOT.TTree):
            # Retrieve the associated file
            treefile = firstarg.GetCurrentFile()
            if not treefile:
                # The tree has no associated input file
                return None
            else:
                return [treefile.GetName()]

        if len(self.args) > 1:
            # Second argument can be:
            # 1. A simple string representing a file or a glob of files
            # 2. A vector of filename strings
            # 3. A pointer to a TDirectory (i.e. the file of the dataset)
            secondarg = self.args[1]
            if isinstance(secondarg, str):
                # Expand globbing excluding remote files
                remote_prefixes = ("root:", "http:", "https:")
                if not secondarg.startswith(remote_prefixes):
                    return glob.glob(secondarg)
                else:
                    return [secondarg]
            elif isinstance(secondarg, (list, ROOT.std.vector("string"))):
                # Make sure this returns a list of Python strings
                return [str(filename) for filename in secondarg]
            elif isinstance(secondarg, ROOT.TDirectory):
                # Return the name of the file
                return [str(secondarg.GetName())]

        return None

    def _get_friend_info(self):
        """
        Retrieve friend tree names and filenames of a given `ROOT.TTree`
        object.

        Args:
            tree (ROOT.TTree): the ROOT.TTree instance used as an argument to
                DistRDF.RDataFrame(). ROOT.TChain inherits from ROOT.TTree so it
                is a valid argument too.

        Returns:
            (FriendInfo): A FriendInfo instance with two lists as variables.
                The first list holds the names of the friend tree(s), the
                second list holds the file names of each of the trees in the
                first list, each tree name can correspond to multiple file
                names.
        """
        friend_names = []
        friend_file_names = []

        # Retrieve the TTree instance
        tree = self.get_tree()

        # Early return if the dataframe wasn't constructed from a TTree.
        if tree is None:
            return FriendInfo()

        # Get a list of ROOT.TFriendElement objects
        friends = tree.GetListOfFriends()
        if not friends:
            # RDataFrame may have been created with a TTree without
            # friend trees.
            return FriendInfo()

        for friend in friends:
            friend_tree = friend.GetTree()  # ROOT.TTree
            real_name = friend_tree.GetName()  # Treename as string

            # TChain inherits from TTree
            if isinstance(friend_tree, ROOT.TChain):
                cur_friend_files = [
                    # The title of a TFile is the file name
                    chain_file.GetTitle()
                    for chain_file
                    # Get a list of ROOT.TFile objects
                    in friend_tree.GetListOfFiles()
                ]

            else:
                cur_friend_files = [
                    friend_tree.
                    GetCurrentFile().  # ROOT.TFile
                    GetName()  # Filename as string
                ]
            friend_file_names.append(cur_friend_files)
            friend_names.append(real_name)

        return FriendInfo(friend_names, friend_file_names)

    def build_ranges(self):
        """Build the ranges for this dataset."""
        # Empty datasets cannot be processed distributedly
        if not self.nentries:
            raise RuntimeError(
                ("Cannot build a distributed RDataFrame with zero entries. "
                 "Distributed computation will fail. "))

        return Ranges.get_clustered_ranges(self.treename, self.inputfiles, self._get_friend_info())
