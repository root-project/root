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

from __future__ import print_function

import collections
import glob
import logging
import warnings

import ROOT

from DistRDF.Operation import Operation

logger = logging.getLogger(__name__)

Range = collections.namedtuple("Range",
                               ["start", "end", "filelist", "friend_info"])


def _n_even_chunks(iterable, n_chunks):
    """
    Yield `n_chunks` as even chunks as possible from `iterable`. Though generic,
    this function is used in _get_clustered_ranges to split a list of clusters
    into multiple sublists. Each sublist will hold the clusters that should fit
    in a single partition of the distributed dataset::

        [
            # Partition 1 will process the following clusters
            [
                (start_0_0, end_0_0, offset_0, (filename_0, 0)),
                (start_0_1, end_0_1, offset_0, (filename_0, 0)),
                ...,
                (start_1_0, end_1_0, offset_1, (filename_1, 1)),
                (start_1_1, end_1_1, offset_1, (filename_1, 1)),
                ...,
                (start_n_0, end_n_0, offset_n, (filename_n, n)),
                (start_n_1, end_n_1, offset_n, (filename_n, n)),
                ...
            ],
            # Partition 2 will process these other clusters
            [
                (start_n+1_0, end_n+1_0, offset_n+1, (filename_n+1, n+1)),
                (start_n+1_1, end_n+1_1, offset_n+1, (filename_n+1, n+1)),
                ...,
                (start_m_0, end_m_0, offset_m, (filename_m, m)),
                (start_m_1, end_m_1, offset_m, (filename_m, m)),
                ...
            ],
            ...
        ]

    """
    last = 0
    itlenght = len(iterable)
    for i in range(1, n_chunks + 1):
        cur = int(round(i * (itlenght / n_chunks)))
        yield iterable[last:cur]
        last = cur


class Node(object):
    """
    A Class that represents a node in RDataFrame operations graph. A Node
    houses an operation and has references to children nodes.
    For details on the types of operations supported, try :

    Example::

        import DistRDF
        DistRDF.use(...) # Choose your backend
        print(DistRDF.current_backend.supported_operations)

    Attributes:
        get_head (function): A lambda function that returns the head node of
            the current graph.

        operation: The operation that this Node represents.
            This could be :obj:`None`.

        children (list): A list of :obj:`DistRDF.Node` objects which represent
            the children nodes connected to the current node.

        _new_op_name (str): The name of the new incoming operation of the next
            child, which is the last child node among the current node's
            children.

        value: The computed value after executing the operation in the current
            node for a particular DistRDF graph. This is permanently :obj:`None`
            for transformation nodes and the action nodes get a
            :obj:`ROOT.RResultPtr` after event-loop execution.

        pyroot_node: Reference to the PyROOT object that implements the
            functionality of this node on the cpp side.

        has_user_references (bool): A flag to check whether the node has
            direct user references, that is if it is assigned to a variable.
            Default value is :obj:`True`, turns to :obj:`False` if the proxy
            that wraps the node gets garbage collected by Python.
    """

    def __init__(self, get_head, operation, *args):
        """
        Creates a new node based on the operation passed as argument.

        Args:
            get_head (function): A lambda function that returns the head node
                of the current graph. This value could be `None`.

            operation (DistRDF.Operation.Operation): The operation that this Node
                represents. This could be :obj:`None`.
        """
        if get_head is None:
            # Function to get 'head' Node
            self.get_head = lambda: self
        else:
            self.get_head = get_head

        self.operation = operation
        self.children = []
        self._new_op_name = ""
        self.value = None
        self.pyroot_node = None
        self.has_user_references = True

    def __getstate__(self):
        """
        Converts the state of the current node
        to a Python dictionary.

        Returns:
            dictionary: A dictionary that stores all instance variables
            that represent the current DistRDF node.

        """
        state_dict = {'children': self.children}
        if self.operation:
            state_dict['operation_name'] = self.operation.name
            state_dict['operation_args'] = self.operation.args
            state_dict['operation_kwargs'] = self.operation.kwargs

        return state_dict

    def __setstate__(self, state):
        """
        Retrieves the state dictionary of the current
        node and sets the instance variables.

        Args:
            state (dict): This is the state dictionary that needs to be
                converted to a `Node` object.

        """
        self.children = state['children']
        if state.get('operation_name'):
            self.operation = Operation(state['operation_name'],
                                       *state['operation_args'],
                                       **state["operation_kwargs"])
        else:
            self.operation = None

    def is_prunable(self):
        """
        Checks whether the current node can be pruned from the computational
        graph.

        Returns:
            bool: True if the node has no children and no user references or
            its value has already been computed, False otherwise.
        """
        if not self.children:
            # Every pruning condition is written on a separate line
            if not self.has_user_references or \
               (self.operation and self.operation.is_action() and self.value):

                # ***** Condition 1 *****
                # If the node is wrapped by a proxy which is not directly
                # assigned to a variable, then it will be flagged for pruning

                # ***** Condition 2 *****
                # If the current node's value was already
                # computed, it should get pruned only if it's
                # an Action node.

                # Logger debug statements
                logger.debug("{} node can be pruned".format(
                    self.operation.name
                ))

                return True

        # Logger debug statements
        if self.operation:  # Node has an operation
            logger.debug("{} node shouldn't be pruned".format(
                self.operation.name
            ))
        else:  # Node is the RDataFrame
            logger.debug("Graph pruning completed")
        return False

    def graph_prune(self):
        """
        Prunes nodes from the current DistRDF graph under certain conditions.
        The current node will be pruned if it has no children and the user
        application does not hold any reference to it. The children of the
        current node will get recursively pruned.

        Returns:
            bool: True if the current node has to be pruned, False otherwise.
        """
        children = []

        # Logger debug statements
        if self.operation:
            logger.debug("Checking {} node for pruning".format(
                self.operation.name
            ))
        else:
            logger.debug("Starting computational graph pruning")

        for n in self.children:
            # Logger debug statement
            # Select children based on pruning condition
            if not n.graph_prune():
                children.append(n)

        self.children = children
        return self.is_prunable()


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


class RangesBuilder(object):

    def __init__(self, headnode):
        """Initialization"""

        self._headnode = headnode

    @property
    def npartitions(self):
        """Retrieve the npartitions value from the head node of the graph."""
        return self._headnode.npartitions

    @npartitions.setter
    def npartitions(self, value):
        """Change the npartitions value of the head node."""
        self._headnode.npartitions = value

    @property
    def nentries(self):
        """Retrieve the total number of entries from the dataset."""
        return self._headnode.get_num_entries()

    @property
    def treename(self):
        """Retrieve the name of the TTree."""
        return self._headnode.get_treename()

    @property
    def tree(self):
        """Retrieve the TTree instance."""
        return self._headnode.get_tree()

    @property
    def inputfiles(self):
        """Retrieve the input files of the dataset."""
        return self._headnode.get_inputfiles()

    @property
    def friendinfo(self):
        """Retrieve information about friend trees of the dataset."""
        return self._headnode._get_friend_info()

    def get_clusters(self, treename, filelist):
        """
        Extract a list of cluster boundaries for the given tree and files

        Args:
            treename (str): Name of the TTree split into one or more files.

            filelist (list): List of one or more ROOT files.

        Returns:
            list: List of tuples defining the cluster boundaries. Each tuple
            contains four elements: first entry of a cluster, last entry of
            cluster (exclusive), offset of the cluster and file where the
            cluster belongs to::

                [
                    (0, 100, 0, ("filename_1.root", 0)),
                    (100, 200, 0, ("filename_1.root", 0)),
                    ...,
                    (10000, 10100, 10000, ("filename_2.root", 1)),
                    (10100, 10200, 10000, ("filename_2.root", 1)),
                    ...,
                    (n, n+100, n, ("filename_n.root", n)),
                    (n+100, n+200, n, ("filename_n.root", n)),
                    ...
                ]
        """

        clusters = []
        cluster = collections.namedtuple(
            "cluster", ["start", "end", "offset", "filetuple"])
        fileandindex = collections.namedtuple("fileandindex",
                                              ["filename", "index"])
        offset = 0
        fileindex = 0

        for filename in filelist:
            f = ROOT.TFile.Open(filename)
            t = f.Get(treename)

            entries = t.GetEntriesFast()
            it = t.GetClusterIterator(0)
            start = it()
            end = 0

            while start < entries:
                end = it()
                clusters.append(cluster(start + offset, end + offset, offset,
                                        fileandindex(filename, fileindex)))
                start = end

            fileindex += 1
            offset += entries

        logger.debug("Returning files with their clusters:\n%s",
                     "\n\n".join(map(str, clusters)))

        return clusters

    def _get_balanced_ranges(self, nentries):
        """
        Builds range pairs from the given values of the number of entries in
        the dataset and number of partitions required. Each range contains the
        same amount of entries, except for those cases where the number of
        entries is not a multiple of the partitions.

        Args:
            nentries (int): The number of entries in a dataset.

        Returns:
            list: List of :obj:`Range`s objects.
        """
        partition_size = int(nentries / self.npartitions)

        i = 0  # Iterator

        ranges = []

        remainder = nentries % self.npartitions

        while i < nentries:
            # Start value of current range
            start = i
            end = i = start + partition_size

            if remainder:
                # If the modulo value is not
                # exhausted, add '1' to the end
                # of the current range
                end = i = end + 1
                remainder -= 1

            ranges.append(Range(start, end, None, None))

        return ranges

    def _get_clustered_ranges(self, treename, filelist,
                              friend_info=None):
        """
        Builds ``Range`` objects taking into account the clusters of the
        dataset. Each range will represent the entries processed within a single
        partition of the distributed dataset.

        Args:
            treename (str): Name of the tree.

            filelist (list): List of ROOT files.

            friend_info (FriendInfo): Information about friend trees.

        Returns:
            list[collections.namedtuple]: List containinig the ranges in which
            the dataset has been split for distributed execution. Each ``Range``
            contains a starting entry, an ending entry, the list of files
            that are traversed to get all the entries and information about
            friend trees::

                [
                    Range(start=0,
                        end=42287856,
                        filelist=['Run2012B_TauPlusX.root',
                                  'Run2012C_TauPlusX.root'],
                        friend_info=None),
                    Range(start=6640348,
                        end=51303171,
                        filelist=['Run2012C_TauPlusX.root'],
                        friend_info=None)
                ]

        """

        # Retrieve a list of clusters for all files of the tree
        clustersinfiles = self.get_clusters(treename, filelist)
        numclusters = len(clustersinfiles)

        # Restrict `npartitions` if it's greater than clusters of the dataset
        if self.npartitions > numclusters:
            msg = ("Number of partitions is greater than number of clusters "
                   "in the dataset. Using {} partition(s)".format(numclusters))
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.npartitions = numclusters

        logger.debug("%s clusters will be split along %s partitions.",
                     numclusters, self.npartitions)

        """
        This list comprehension builds ``Range`` tuples with the following
        elements:

        1. ``start``: The minimum entry among all the clusters considered in a
           given partition. The offset of the first cluster of the list is
           subtracted. This is useful to keep the reference of the range with
           respect to the current files (see below).
        2. ``end``: The maximum entry among all the clusters considered in a
           given partition. The offset of the first cluster of the list is
           subtracted. This is useful to keep the reference of the range with
           respect to the current files (see below).
        3. ``filelist``: The list of files that are span between entries
           ``start`` and ``end``::

                Filelist: [file_1,file_2,file_3,file_4]

                Clustered range: [0,150]
                file_1 holds entries [0, 100]
                file_2 holds entries [101, 200]
                Then the clustered range should open [file_1, file_2]

                Clustered range: [150,350]
                file_3 holds entris [201, 300]
                file_4 holds entries [301, 400]
                Then the clustered range should open [file_2, file_3, file_4]

           Each ``cluster`` namedtuple has a ``fileandindex`` namedtuple. The
           second element of this tuple corresponds to the index of the file in
           the input `TChain`. This way all files can be uniquely identified,
           even if there is some repetition (e.g. when building a TChain with
           multiple instances of the same file). The algorithm to retrieve the
           correct files for each range takes the unique filenames from the list
           of clusters and sorts them by their index to keep the original order.

           In each file only the clusters needed to process the clustered range
           will be read.
        4. ``friend_info``: Information about friend trees.

        In each range, the offset of the first file is always subtracted to the
        ``start`` and ``end`` entries. This is needed to maintain a reference of
        the entries of the range with respect to the list of files that hold
        them. For example, given the following files::

            tree10000entries10clusters.root --> 10000 entries, 10 clusters
            tree20000entries10clusters.root --> 20000 entries, 10 clusters
            tree30000entries10clusters.root --> 30000 entries, 10 clusters

        Building 2 ranges will lead to the following tuples::

            Range(start=0,
                  end=20000,
                  filelist=['tree10000entries10clusters.root',
                            'tree20000entries10clusters.root'],
                  friend_info=None)

            Range(start=10000,
                  end=50000,
                  filelist=['tree20000entries10clusters.root',
                            'tree30000entries10clusters.root'],
                  friend_info=None)

        The first ``Range`` will read the first 10000 entries from the first
        file, then switch to the second file and read the first 10000 entries.
        The second ``Range`` will start from entry number 10000 of the second
        file up until the end of that file (entry number 20000), then switch to
        the third file and read the whole 30000 entries there.
        """
        clustered_ranges = [
            Range(
                min(clusters)[0] - clusters[0].offset,  # type: int
                max(clusters)[1] - clusters[0].offset,  # type: int
                [
                    filetuple.filename
                    for filetuple in sorted(set([
                        cluster.filetuple for cluster in clusters
                    ]), key=lambda curtuple: curtuple[1])
                ],  # type: list[str]
                friend_info  # type: FriendInfo
            )  # type: collections.namedtuple
            for clusters in _n_even_chunks(clustersinfiles, self.npartitions)
        ]

        logger.debug("Created following clustered ranges:\n%s",
                     "\n\n".join(map(str, clustered_ranges)))

        return clustered_ranges

    def build_ranges(self):
        """
        Define two type of ranges based on the arguments passed to the
        RDataFrame head node.
        """
        # Create variables here to call properties only once
        nentries = self.nentries
        treename = self.treename
        inputfiles = self.inputfiles
        friendinfo = self.friendinfo

        # Empty trees cannot be processed distributedly
        if not nentries:
            raise RuntimeError(
                ("No entries in the TTree. "
                 "Distributed computation will fail. "
                 "Please make sure your dataset is not empty."))

        if self.npartitions > nentries:
            # Restrict 'npartitions' if it's greater than 'nentries'
            self.npartitions = nentries

        if treename is not None:
            if inputfiles is None:
                raise RuntimeError(("Found tree with name {} but no files associated with it. "
                                    "Make sure your dataset is saved to disk first."))
            logger.debug("Building clustered ranges for tree %s with the "
                         "following input files:\n%s",
                         treename,
                         list(inputfiles)
                         )
            return self._get_clustered_ranges(treename, inputfiles, friendinfo)
        else:
            logger.debug(
                "Building balanced ranges for %d entries.", nentries)
            return self._get_balanced_ranges(nentries)


class HeadNode(Node):
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
        super(HeadNode, self).__init__(None, None, *args)
        # Early check of the arguments to RDataFrame constructor
        df_check_args = ROOT.RDataFrame(*args)
        self.args = list(args)  # Make args mutable

        # Set at creation of the dataframe, might be optimized by the backend
        # in optimize_partitions
        self.npartitions = 2

    def build_ranges(self):
        """Build the ranges for this dataset."""
        return RangesBuilder(self).build_ranges()

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
        first_arg = self.args[0]
        if isinstance(first_arg, int):
            # If there's only one argument
            # which is an integer, return it.
            return first_arg
        elif isinstance(first_arg, ROOT.TTree):
            # If the argument is a TTree or TChain,
            # get the number of entries from it.
            return first_arg.GetEntries()

        second_arg = self.args[1]

        # Construct a ROOT.TChain object
        chain = ROOT.TChain(first_arg)

        if isinstance(second_arg, str):
            # If the second argument is a string
            chain.Add(second_arg)
        else:
            # If the second argument is a list or vector
            for fname in second_arg:
                chain.Add(str(fname))  # Possible bug in conversion of string

        return chain.GetEntries()

    def get_treename(self):
        """
        Get name of the TTree.

        Returns:
            (str, None): Name of the TTree, or :obj:`None` if there is no tree.

        """
        first_arg = self.args[0]
        if isinstance(first_arg, ROOT.TTree):
            # Get name from a given TTree/TChain
            return first_arg.GetName()
        elif isinstance(first_arg, str):
            # First argument was the name of the tree
            return first_arg
        # RDataFrame may have been created without any TTree or TChain
        return None

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
                raise RuntimeError(("Trees with no associated files are not supported. "
                                    "Please save your tree to a file and retry."))
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

        # Since the dataset could be just an empty dataframe with some entries,
        # aka RDataFrame(ULong64_t numEntries), we still have to return None
        # here. Would be nice to have separate head nodes for different data
        # sources to better separate the roles.
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
