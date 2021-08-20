import itertools
import logging

from functools import total_ordering

import ROOT

logger = logging.getLogger(__name__)


class EmptySourceRange(object):
    """
    Empty source range of entries

    Attributes:

    id (int): Sequential counter to identify this range. It is used to assign
        a filename to a partial Snapshot in case it was requested. The id is
        assigned in `get_balanced_ranges` to ensure that each distributed
        RDataFrame run has a list of ranges with sequential ids starting from
        zero.

    start (int): Starting entry of this range.

    end (int): Ending entry of this range.
    """

    def __init__(self, rangeid, start, end):
        """set attributes"""
        self.id = rangeid
        self.start = start
        self.end = end

class TreeRange(object):
    """
    TTree range of entries. The entries are local with respect to the list of
    files that are processed with this range. These files are a subset of the
    global list of input files of the original dataset.

    Attributes:

    id (int): Sequential counter to identify this range. It is used to assign
        a filename to a partial Snapshot in case it was requested. The id is
        assigned in `get_clustered_ranges` to ensure that each distributed
        RDataFrame run has a list of ranges with sequential ids starting from
        zero.

    globalstart (int): Starting entry relative to the TChain made with the list
        of files of this range.

    globalend (int): Ending entry relative to the TChain made with the list
        of files of this range.

    localstarts (list[int]): List of starting entries relative to each single
        file in this range.

    localends (list[int]): List of ending entries relative to each single
        file in this range.

    filelist (list[str]): List of files to be processed with this range.

    treesnentries (list[int]): List of total number of entries relative to each
        single file in this range.

    friendinfo (ROOT.Internal.TreeUtils.RFriendInfo, None): Information about
        friend trees of the chain built for this range. Not None if the user
        provided a TTree or TChain in the distributed RDataFrame constructor.
    """

    def __init__(self, rangeid, globalstart, globalend, localstarts, localends, filelist, treesnentries, friendinfo):
        """set attributes"""
        self.id = rangeid
        self.globalstart = globalstart
        self.globalend = globalend
        self.localstarts = localstarts
        self.localends = localends
        self.filelist = filelist
        self.treesnentries = treesnentries
        self.friendinfo = friendinfo


@total_ordering
class FileAndIndex(object):
    """
    This is a pair (filename, index) that represents the index of the current
    filename in the list of input files of the dataset

    Attributes:

    filename (str): The name of the file.

    index (int): The index of the file in the list of input files.
    """

    def __init__(self, filename, fileindex):
        """set attributes"""
        self.filename = filename
        self.fileindex = fileindex

    def __lt__(self, other):
        """Total ordering is defined to be able to sort a set[FileAndIndex]."""
        return self.fileindex < other.fileindex

    def __eq__(self, other):
        """Defined in compliance with total_ordering decorator"""
        return self.fileindex == other.fileindex

    def __hash__(self):
        """
        This type needs to be hashable to be part of a set in
        get_clustered_ranges
        """
        return hash(self.filename)


@total_ordering
class ChainCluster(object):
    """
    Descriptor of a cluster of entries in a TChain. Represents entries local to
    the current file in the chain.

    Attributes:

    start (int): The starting file-local entry of this cluster in the chain.

    end (int): The ending file-local entry of this cluster in the chain.

    offset (int): The offset of this cluster in the chain. That is, the amount
        of entries seen in the chain up to the beginning of the file this
        cluster belongs to.

    filetuple (FileAndIndex): A pair with the name of the file this cluster
        belongs to and the index of that file in the chain.

    treenentries (int): The total number of entries in the tree this file
        belongs to.
    """

    def __init__(self, start, end, offset, filetuple, treenentries):
        """set attributes"""
        self.start = start
        self.end = end
        self.offset = offset
        self.filetuple = filetuple
        self.treenentries = treenentries

    def __lt__(self, other):
        """
        In `get_clustered_ranges` we need to retrieve the minimum and maximum
        entries in a certain list of clusters.
        """
        return self.start < other.start and self.end < other.end

    def __eq__(self, other):
        """Defined in compliance with total_ordering decorator"""
        return (self.start == other.start and
               self.end == other.end and
               self.offset == other.offset and
               self.filetuple.filename == other.filetuple.filename and
               self.filetuple.fileindex == other.filetuple.fileindex)


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


def get_clusters(treename, filelist):
    """
    Extract a list of cluster boundaries for the given tree and files

    Args:
        treename (str): Name of the TTree split into one or more files.
        filelist (list): List of one or more ROOT filenames.

    Returns:
        list[ChainCluster]: Tuples with cluster boundaries and some metadata.
        For example::

            [
                (0, 100, 0, ("filename_1.root", 0), 10000),
                (100, 200, 0, ("filename_1.root", 0), 10000),
                ...,
                (10000, 10100, 10000, ("filename_2.root", 1), 10000),
                (10100, 10200, 10000, ("filename_2.root", 1), 10000),
                ...,
                (n, n+100, n, ("filename_n.root", n), 10000),
                (n+100, n+200, n, ("filename_n.root", n), 10000),
                ...
            ]

    """

    clusters = []

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
            clusters.append(ChainCluster(start, end, offset,
                                    FileAndIndex(filename, fileindex), entries))
            start = end

        fileindex += 1
        offset += entries

    logger.debug("Returning files with their clusters:\n%s",
                 "\n\n".join(map(str, clusters)))

    return clusters


def get_balanced_ranges(nentries, npartitions):
    """
    Builds range pairs from the given values of the number of entries in
    the dataset and number of partitions required. Each range contains the
    same amount of entries, except for those cases where the number of
    entries is not a multiple of the partitions.

    Args:
        nentries (int): The number of entries in a dataset.

        npartitions (int): The number of partititions the sequence of entries
            should be split in.

    Returns:
        list[DistRDF.Ranges.EmptySourceRange]: Each element of the list contains
            the start and end entry of the corresponding range.
    """

    partition_size = nentries // npartitions

    i = 0  # Iterator

    ranges = []

    remainder = nentries % npartitions

    rangeid = 0 # Keep track of the current range id
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

        ranges.append(EmptySourceRange(rangeid, start, end))
        rangeid += 1

    return ranges


def get_clustered_ranges(clustersinfiles, npartitions, friendinfo):
    """
    Builds ``TreeRange`` objects taking into account the clusters of the
    dataset. Each range will represent the entries processed within a single
    partition of the distributed dataset.

    Args:
        clustersinfiles (list): List of namedtuples representing clusters in
            the input files of the current dataset.

        npartitions (int): Number of ranges that will be produced.

        friendinfo (ROOT.Internal.TreeUtils.RFriendInfo): Information about friend
            trees.

    Returns:
        list[DistRDF.Ranges.TreeRange]: Each element of the list represents one
            range in which the dataset has been split for distributed execution.
            Example output::

            [
                TreeRange(id=0,
                    globalstart=0,
                    globalend=150,
                    localstarts=[0, 0],
                    localends=[50, 100],
                    filelist=['file1.root', 'file2.root'],
                    treesnentries=[50, 200],
                    friendinfo=None),
                TreeRange(id=1,
                    globalstart=150,
                    globalend=300,
                    localstarts=[100, 0],
                    localends=[200, 50],
                    filelist=['file2.root', 'file3.root'],
                    treesnentries=[200, 50],
                    friendinfo=None),
            ]

    """

    clustersbypartition = _n_even_chunks(clustersinfiles, npartitions)


    clustered_ranges = []
    rangeid = 0 # Keep track of the current range id
    for partition in clustersbypartition:

        # One partition looks like:
        #     [
        #         (start_0_0, end_0_0, offset_0, (filename_0, 0), nentries_0),
        #         (start_0_1, end_0_1, offset_0, (filename_0, 0), nentries_0),
        #         ...,
        #         (start_1_0, end_1_0, offset_1, (filename_1, 1), nentries_1),
        #         (start_1_1, end_1_1, offset_1, (filename_1, 1), nentries_1),
        #         ...,
        #         (start_n_0, end_n_0, offset_n, (filename_n, n), nentries_n),
        #         (start_n_1, end_n_1, offset_n, (filename_n, n), nentries_n),
        #         ...
        #     ],
        # We need to retrieve:
        # 1. The `globalstart` and `globalend` entries "global" to the chain
        #    made with the files of this partition.
        # 2. The `localstarts` and `localends` lists, the elements of which are
        #    start and end entries of a set of clusters in a certain file of the
        #    filelist of this partition.
        # 3. The total number of entries for each file in the partition.

        # Let's start with the start and end entries local to each file. We
        # group the current partition by the fileindex, so that we can process
        # all the clusters belonging to the same file together.
        localstarts = []
        localends = []
        filelist = []
        treesnentries = []

        for _, clustersinsamefileiter in itertools.groupby(partition, lambda cluster: cluster.filetuple.fileindex):
            # Grab a list of clusters belonging to the same file to give as
            # argument to min and max
            clustersinsamefilelist = list(clustersinsamefileiter)

            localstarts.append(min(clustersinsamefilelist).start)
            localends.append(max(clustersinsamefilelist).end)
            filelist.append(clustersinsamefilelist[0].filetuple.filename)
            treesnentries.append(clustersinsamefilelist[0].treenentries)

        # The global start and end entries are retrieved as follows:
        # - Take as start start the `start` attribute of the first
        #   `ChainCluster` object in the current partition
        # - Take as end the `end` attribute of the last `ChainCluster` object
        #   in the current partition
        # - Both values need to be shifted by the respective `offset` attribute
        #   in the same `ChainCluster` instance
        # - Then to make them "global with respect to the chain of files in the
        #   current partition", we need to subtract the offset of the first
        #   cluster in the current partition. That is equal to the amount of
        #   entries in the chain up to the beginning of the file that cluster
        #   belongs to. This is needed to maintain a reference of the entries of
        #   the range with respect to the list of files that hold them. For
        #   example, given the following files:
        #       tree10000entries10clusters.root --> 10000 entries, 10 clusters
        #       tree20000entries10clusters.root --> 20000 entries, 10 clusters
        #       tree30000entries10clusters.root --> 30000 entries, 10 clusters
        #   Building 2 ranges will lead to the following tuples:
        #       TreeRange(id=0,
        #                 globalstart=0,
        #                 globalend=20000,
        #                 localstarts=[0, 0],
        #                 localends=[10000, 10000],
        #                 filelist=['tree10000entries10clusters.root',
        #                           'tree20000entries10clusters.root'],
        #                 treesnentries=[10000, 20000],
        #                 friendinfo=None)
        #       TreeRange(id=1,
        #                 globalstart=10000,
        #                 globalend=50000,
        #                 localstarts=[10000, 0],
        #                 localends=[20000, 30000],
        #                 filelist=['tree20000entries10clusters.root',
        #                           'tree30000entries10clusters.root'],
        #                 treesnentries=[20000,30000],
        #                 friendinfo=None)
        #   The first `TreeRange` will read the first 10000 entries from the
        #   first file, then switch to the second file and read the first 10000
        #   entries. The second `TreeRange` will start from entry number 10000
        #   of the second file up until the end of that file (entry number
        #   20000), then switch to the third file and read the whole 30000
        #   entries there.
        firstclusterinpartition = partition[0]
        partitionoffset = firstclusterinpartition.offset

        lastclusterinpartition = partition[-1]

        # Cluster offsets are relative to the offset of the first cluster in the
        # partition. For the first cluster, this would mean adding and
        # subtracting the same quantity, no need to do that.
        globalstart = firstclusterinpartition.start
        globalend = lastclusterinpartition.end + lastclusterinpartition.offset - partitionoffset

        clustered_ranges.append(TreeRange(rangeid, globalstart, globalend, localstarts, localends, filelist, treesnentries, friendinfo))
        rangeid += 1

    return clustered_ranges
