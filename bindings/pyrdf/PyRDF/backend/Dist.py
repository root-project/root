from __future__ import print_function

import collections
import glob
import logging
import warnings
from abc import abstractmethod

import numpy
import PyRDF
import ROOT
from PyRDF.backend import Backend

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


class Dist(Backend.Backend):
    """
    Base class for implementing all distributed backends.

    Attributes:
        npartitions (int): The number of chunks to divide the dataset in, each
            chunk is then processed in parallel.

        supported_operations (list): list of supported RDataFrame operations
            in a distributed environment.

        friend_info (PyRDF.Dist.FriendInfo): A class instance that holds
            information about any friend trees of the main ROOT.TTree
    """

    def __init__(self, config={}):
        """
        Creates an instance of Dist.

        Args:
            config (dict, optional): The config options for the current
                distributed backend. Default value is an empty python
                dictionary: :obj:`{}`.

        """
        super(Dist, self).__init__(config)
        # Operations that aren't supported in distributed backends
        operations_not_supported = [
            'Mean',
            'Max',
            'Min',
            'Range',
            'Take',
            'Foreach',
            'Reduce',
            'Report',
            'Aggregate'
        ]

        # Remove the value of 'npartitions' from config dict
        self.npartitions = config.pop('npartitions', None)

        self.supported_operations = [op for op in self.supported_operations
                                     if op not in operations_not_supported]

        self.friend_info = FriendInfo()

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
            f = ROOT.TFile.Open(str(filename))
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
                              friend_info=FriendInfo()):
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

    def _get_filelist(self, files):
        """
        Convert single file into list of files and expand globbing

        Args:
            files (str, list): String containing name of a single file or list
                with several file names, both cases may contain globbing
                characters.

        Returns:
            list: list of file names.
        """
        if isinstance(files, str):
            # Expand globbing excluding remote files
            remote_prefixes = ("root:", "http:", "https:")
            if not files.startswith(remote_prefixes):
                files = glob.glob(files)
            else:
                # Convert single file into a filelist
                files = [files, ]

        return files

    def build_ranges(self):
        """
        Define two type of ranges based on the arguments passed to the
        RDataFrame head node.
        """
        if self.npartitions > self.nentries:
            # Restrict 'npartitions' if it's greater
            # than 'nentries'
            self.npartitions = self.nentries

        if self.treename and self.files:
            filelist = self._get_filelist(self.files)
            logger.debug("Building clustered ranges for tree %s with the "
                         "following input files:\n%s",
                         self.treename,
                         list(self.files)
                         )
            return self._get_clustered_ranges(self.treename, filelist,
                                              self.friend_info)
        else:
            logger.debug(
                "Building balanced ranges for %d entries.", self.nentries)
            return self._get_balanced_ranges(self.nentries)

    def _get_friend_info(self, tree):
        """
        Retrieve friend tree names and filenames of a given `ROOT.TTree`
        object.

        Args:
            tree (ROOT.TTree): the ROOT.TTree instance used as an argument to
                PyRDF.RDataFrame(). ROOT.TChain inherits from ROOT.TTree so it
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

    def execute(self, generator):
        """
        Executes the current RDataFrame graph
        in the given distributed environment.

        Args:
            generator (PyRDF.CallableGenerator): An instance of
                :obj:`CallableGenerator` that is responsible for generating
                the callable function.
        """
        callable_function = generator.get_callable()
        # Arguments needed to create PyROOT RDF object
        rdf_args = generator.head_node.args
        treename = generator.head_node.get_treename()
        selected_branches = generator.head_node.get_branches()

        # Avoid having references to the instance inside the mapper
        initialization = Backend.Backend.initialization

        def mapper(current_range):
            """
            Triggers the event-loop and executes all
            nodes in the computational graph using the
            callable.

            Args:
                current_range (tuple): A pair that contains the starting and
                    ending values of the current range.

            Returns:
                list: This respresents the list of values of all action nodes
                in the computational graph.
            """
            import ROOT

            # We have to decide whether to do this in Dist or in subclasses
            # Utils.declare_headers(worker_includes)  # Declare headers if any
            # Run initialization method to prepare the worker runtime
            # environment
            initialization()

            # Build rdf
            start = int(current_range.start)
            end = int(current_range.end)

            if treename:
                # Build TChain of files for this range:
                chain = ROOT.TChain(treename)
                for f in current_range.filelist:
                    chain.Add(str(f))

                # We assume 'end' is exclusive
                chain.SetCacheEntryRange(start, end)

                # Gather information about friend trees
                friend_info = current_range.friend_info
                if friend_info:
                    # Zip together the treenames of the friend trees and the
                    # respective file names. Each friend treename can have
                    # multiple corresponding friend file names.
                    tree_files_names = zip(
                        friend_info.friend_names,
                        friend_info.friend_file_names
                    )
                    for friend_treename, friend_filenames in tree_files_names:
                        # Start a TChain with the current friend treename
                        friend_chain = ROOT.TChain(friend_treename)
                        # Add each corresponding file to the TChain
                        for filename in friend_filenames:
                            friend_chain.Add(filename)

                        # Set cache on the same range as the parent TChain
                        friend_chain.SetCacheEntryRange(start, end)
                        # Finally add friend TChain to the parent
                        chain.AddFriend(friend_chain)

                if selected_branches:
                    rdf = ROOT.ROOT.RDataFrame(chain, selected_branches)
                else:
                    rdf = ROOT.ROOT.RDataFrame(chain)
            else:
                rdf = ROOT.ROOT.RDataFrame(*rdf_args)  # PyROOT RDF object

            # # TODO : If we want to run multi-threaded in a Spark node in
            # # the future, use `TEntryList` instead of `Range`
            # rdf_range = rdf.Range(current_range.start, current_range.end)

            # Output of the callable
            output = callable_function(rdf, rdf_range=current_range)

            for i in range(len(output)):
                # `AsNumpy` and `Snapshot` return respectively `dict` and `list`
                # that don't have the `GetValue` method.
                if isinstance(output[i], (dict, list)):
                    continue
                # FIX ME : RResultPtrs aren't serializable,
                # because of which we have to manually find
                # out the types here and copy construct the
                # values.

                # The type of the value of the action node
                value_type = type(output[i].GetValue())
                # The `value_type` is required here because,
                # after a call to `GetValue`, the values die
                # along with the RResultPtrs
                output[i] = value_type(output[i].GetValue())
            return output

        def reducer(values_list1, values_list2):
            """
            Merges two given lists of values that were
            returned by the mapper function for two different
            ranges.

            Args:
                values_list1 (list): A list of computed values for a given
                    entry range in a dataset.

                values_list2 (list): A list of computed values for a given
                    entry range in a dataset.

            Returns:
                list: This is a list of values obtained after merging two
                given lists.
            """
            import ROOT

            for i in range(len(values_list1)):
                # A bunch of if-else conditions to merge two values

                # Create a global list with all the files of the partial
                # snapshots
                if isinstance(values_list1[i], list):
                    values_list1[i].extend(values_list2[i])

                elif isinstance(values_list1[i], dict):
                    combined = {
                        key: numpy.concatenate([values_list1[i][key],
                                                values_list2[i][key]])
                        for key in values_list1[i]
                    }
                    values_list1[i] = combined
                elif (isinstance(values_list1[i], ROOT.TH1) or
                      isinstance(values_list1[i], ROOT.TH2)):
                    # Merging two objects of type ROOT.TH1D or ROOT.TH2D
                    values_list1[i].Add(values_list2[i])

                elif isinstance(values_list1[i], ROOT.TGraph):
                    # Prepare a TList
                    tlist = ROOT.TList()
                    tlist.Add(values_list2[i])

                    # Merge the second graph onto the first
                    num_points = values_list1[i].Merge(tlist)

                    # Check if there was an error in merging
                    if num_points == -1:
                        msg = "Error reducing two result values of type TGraph!"
                        raise Exception(msg)

                elif isinstance(values_list1[i], float):
                    # Adding values resulting from a Sum() operation
                    # Sum() always returns a float in python
                    values_list1[i] += values_list2[i]

                elif (isinstance(values_list1[i], int) or
                        isinstance(values_list1[i], long)):  # noqa: Python 2
                    # Adding values resulting from a Count() operation
                    values_list1[i] += values_list2[i]

                else:
                    msg = ("Type \"{}\" is not supported by the reducer yet!"
                           .format(type(values_list1[i])))
                    raise NotImplementedError(msg)

            return values_list1

        # Get number of entries in the input dataset using
        # arguments passed to RDataFrame constructor
        self.nentries = generator.head_node.get_num_entries()

        # Retrieve the treename used to initialize the RDataFrame
        self.treename = generator.head_node.get_treename()

        # Retrieve the filenames used to initialize the RDataFrame
        self.files = generator.head_node.get_inputfiles()

        # Retrieve the ROOT.TTree instance used to initialize the RDataFrame
        self.tree = generator.head_node.get_tree()

        # Retrieve info about the friend trees
        if self.tree:
            self.friend_info = self._get_friend_info(self.tree)

        if not self.nentries:
            # Fall back to local execution
            # if 'nentries' is '0'
            msg = ("No entries in the Tree, falling back to local execution!")
            warnings.warn(msg, UserWarning, stacklevel=2)
            PyRDF.use("local")
            from .. import current_backend
            return current_backend.execute(generator)

        # Values produced after Map-Reduce
        values = self.ProcessAndMerge(mapper, reducer)
        # List of action nodes in the same order as values
        nodes = generator.get_action_nodes()

        # Set the value of every action node
        for node, value in zip(nodes, values):
            if node.operation.name == "Snapshot":
                # Retrieve treename from operation args and start TChain
                snapshot_treename = node.operation.args[0]
                snapshot_chain = ROOT.TChain(snapshot_treename)
                # Add partial snapshot files to the chain
                for filename in value:
                    snapshot_chain.Add(filename)
                # Create a new rdf with the chain and return that to user
                snapshot_rdf = PyRDF.RDataFrame(snapshot_chain)
                node.value = snapshot_rdf
            else:
                node.value = value

    @abstractmethod
    def ProcessAndMerge(self, mapper, reducer):
        """
        Subclasses must define how to run map-reduce functions on a given
        backend.
        """
        pass

    @abstractmethod
    def distribute_files(self, includes_list):
        """
        Subclasses must define how to send all files needed for the analysis
        (like headers and libraries) to the workers.
        """
        pass
