import logging

from bisect import bisect_left
from math import floor
from typing import List, NamedTuple, Optional, Tuple, Union

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

    rangeid = 0  # Keep track of the current range id
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


class TreeRange(NamedTuple):
    """
    Range of entries in one of the trees in the chain of a single distributed task.

    The entries are local with respect to the list of files that are processed
    in this range. These files are a subset of the global list of input files of
    the original dataset.

    Attributes:

    id: Sequential counter to identify this range. It is used to assign a
        filename to a partial Snapshot in case it was requested.

    treenames: List of tree names.

    filenames: List of files to be processed with this range.

    treesnentries: List of total number of entries relative to each tree in this
        range.

    localstarts: List of starting entries relative to each tree in this range.

    localends: List of ending entries relative to each tree in this range.

    globalstart: Starting entry relative to the TChain made with the trees in
        this range.

    globalend: Ending entry relative to the TChain made with the trees in this
        range.

    friendinfo: Information about friend trees of the chain built for this
        range. Not None if the user provided a TTree or TChain in the
        distributed RDataFrame constructor.
    """

    id: int
    treenames: List[str]
    filenames: List[str]
    treesnentries: List[int]
    localstarts: List[int]
    localends: List[int]
    globalstart: int
    globalend: int
    friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo]


class TreeRangePerc(NamedTuple):
    """
    Range of percentages to be considered for a list of trees. Building block
    for an actual range of entries of a distributed task.
    """
    id: int
    treenames: List[str]
    filenames: List[str]
    first_tree_start_perc: int
    last_tree_end_perc: int
    friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo]


def get_clusters_and_entries(treename: str, filename: str) -> Union[Tuple[List[int], int], Tuple[None, None]]:
    """
    Retrieve cluster boundaries and number of entries of a TTree. If the tree
    is empty, returns None, None.
    """

    tfile = ROOT.TFile.Open(filename)
    if not tfile or tfile.IsZombie():
        raise RuntimeError(f"Error opening file '{filename}'.")
    ttree = tfile.Get(treename)

    entries: int = ttree.GetEntriesFast()

    it = ttree.GetClusterIterator(0)
    cluster_startentry: int = it()
    clusters: List[int] = [cluster_startentry]

    while cluster_startentry < entries:
        cluster_startentry = it()
        clusters.append(cluster_startentry)

    tfile.Close()

    if not entries or not clusters:
        return None, None

    return clusters, entries


def get_percentage_ranges(treenames: List[str], filenames: List[str], npartitions: int,
                          friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo]) -> List[TreeRangePerc]:
    """
    Create a list of tasks that will process the given trees partitioning them
    by percentages.
    """
    nfiles = len(filenames)
    files_per_partition = nfiles / npartitions
    # Given a number of files, partition them in npartitions, considering each
    # file as splittable in percentages [0, 1]. Gather:
    # 1. A list of percentages according to how many partitions are required.
    # 2. The corresponding list of file boundaries, as integers.
    # 3. The difference between the two above, to know to which percentage of
    #    a specific file any element of the first list belongs.
    # Example with nfiles = 10 and npartitions = 7
    # percentages = [0., 1.428, 2.857, 4.285, 5.714, 7.142, 8.571, 10.]
    # files_of_percentages = [0, 1, 2, 4, 5, 7, 8, 10]
    # percentages_wrt_files = [0., 0.428, 0.857, 0.285, 0.714, 0.142, 0.571, 0.]
    percentages = [files_per_partition * i for i in range(npartitions+1)]
    files_of_percentages = [floor(percentage) for percentage in percentages]
    percentages_wrt_files = [perc - file for perc, file in zip(percentages, files_of_percentages)]

    # Compute which files are to be considered for the various tasks
    # The indexes of starting files in each task are simply the list of files
    # from above, except for the last value which corresponds to the end of the
    # last file. Also, they are inclusive.
    start_sample_idxs = files_of_percentages[:-1]
    # The indexes of ending files in each task depend on what is the percentage
    # considered for that file. Also, they are exclusive. When the percentage is
    # zero, i.e. we are at a file boundary, we want to consider the whole
    # (previous) file, we just take the file index (shifting the list by one).
    # When the percentage is above zero, we increase the index (shifted by one)
    # by one to be able to consider also the current file.
    end_sample_idxs = [
        file_index + 1 if perc > 0 else file_index
        for file_index, perc in zip(files_of_percentages[1:], percentages_wrt_files[1:])
    ]

    # With the indexes created above, we can partition the lists of names of
    # files and trees. Each task will get a number of trees dictated by the
    # starting index (inclusive) and the ending index (exclusive)
    tasktreenames = [treenames[s:e] for s, e in zip(start_sample_idxs, end_sample_idxs)]
    taskfilenames = [filenames[s:e] for s, e in zip(start_sample_idxs, end_sample_idxs)]

    # Compute the starting percentage of the first tree and the ending percentage
    # of the last tree in each task.
    first_tree_start_perc_tasks = percentages_wrt_files[:-1]
    # When computing the ending percentages, if the percentage defined above is
    # zero, i.e. we are at file boundary, we want to consider the whole tree,
    # thus we set it to one.
    last_tree_end_perc_tasks = [perc if perc > 0 else 1 for perc in percentages_wrt_files[1:]]

    return [
        TreeRangePerc(rangeid, tasktreenames[rangeid], taskfilenames[rangeid],
                      first_tree_start_perc_tasks[rangeid], last_tree_end_perc_tasks[rangeid], friendinfo)
        for rangeid in range(npartitions)
    ]


def get_clustered_range_from_percs(percrange: TreeRangePerc) -> Optional[TreeRange]:
    """
    Builds a range of entries to be processed for each tree in the chain created
    in a distributed task.
    """

    treenames: List[str] = []
    filenames: List[str] = []
    treesnentries: List[int] = []
    localstarts: List[int] = []
    localends: List[int] = []

    # Accumulate entries seen so far in the chain of this task, for:
    # - Deciding whether this task has anything to do (zero entries means empty task).
    # - Computing the ending entry w.r.t. the chain of this task.
    chain_entries_so_far: int = 0

    nfiles = len(percrange.filenames)

    # The starting entry w.r.t. to the chain created in this task. It is always
    # zero, except if the first file in the list contributes to the task and it
    # doesn't also start from zero.
    globalstart = 0

    # Lists of (start, end) percentages to be considered for each tree. The
    # first starting percentage and the last ending percentage are stored in the
    # input percrange object.
    treepercstarts: List[int] = [0] * nfiles
    treepercends: List[int] = [1] * nfiles
    treepercstarts[0] = percrange.first_tree_start_perc
    treepercends[-1] = percrange.last_tree_end_perc

    # Entries in the last tree of this task that is actually processed (i.e.
    # that is not empty).
    last_available_entries: int = 0

    for file_n, (treename, filename, thistreepercstart, thistreepercend) in enumerate(
            zip(percrange.treenames, percrange.filenames, treepercstarts, treepercends)):

        # clusters list contains cluster boundaries of the current file, e.g.:
        # It is important that both the initial entry (0) as well as the last
        # cluster boundary are included in the list. Example:
        # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        clusters, entries = get_clusters_and_entries(treename, filename)
        if entries is None:
            # The tree is empty.
            continue

        # Estimate starting and ending entries that this task has to process
        # in this tree.
        startentry = int(thistreepercstart * entries)
        endentry = int(thistreepercend * entries)

        # Find the corresponding clusters for the above values.
        # The startcluster index is inclusive. bisect_left returns the index
        # corresponding to the correct starting cluster only if startentry is
        # exactly at the cluster boundary. The endcluster index is exclusive.
        # This logic relies on the specific representation of the list of
        # clusters that includes the initial entry (0) as well as the last
        # cluster boundary.
        # Examples:
        # cluster 1: [10, 20]
        # cluster 2: [20, 30]
        # startentry = 10, endentry = 13 --> startcluster = 1, endcluster = 2
        # startentry = 13, endentry = 16 --> startcluster = 2, endcluster = 2
        # startentry = 16, endentry = 19 --> startcluster = 2, endcluster = 2
        # startentry = 19, endentry = 22 --> startcluster = 2, endcluster = 3
        startcluster = bisect_left(clusters, startentry)
        endcluster = bisect_left(clusters, endentry)
        # Avoid creating tasks that will do nothing
        if startcluster == endcluster:
            continue

        tree_startentry_at_cluster_boundary = clusters[startcluster]
        tree_endentry_at_cluster_boundary = clusters[endcluster]

        treenames.append(treename)
        filenames.append(filename)
        treesnentries.append(entries)

        localstarts.append(tree_startentry_at_cluster_boundary)
        localends.append(tree_endentry_at_cluster_boundary)

        if file_n == 0:
            # If we reach this point with the first file in the list, then
            # compute the starting entry global w.r.t. the chain created in this
            # task. It corresponds to the starting entry of the first cluster
            # considered for the tree contained in this file.
            globalstart = tree_startentry_at_cluster_boundary

        chain_entries_so_far += entries
        last_available_entries = entries

    if chain_entries_so_far == 0:
        # No chain should be constructed in this task. This can happen:
        # - If all trees assigned to this task are empty.
        # - If the computed starting cluster is equal to the ending cluster.
        # These would effectively lead to creating a TChain with zero usable
        # entries.
        return None

    # The ending entry w.r.t. the chain of this task is defined as:
    # The total amount of entries in the chain minus the entries of the last tree
    # in the chain plus the ending entry of the last cluster considered for the
    # same tree. We need the first difference to put ourselves at the beginning
    # of the last tree, then just adding up how many entries are actually
    # processed in that tree.
    globalend = chain_entries_so_far - last_available_entries + tree_endentry_at_cluster_boundary

    return TreeRange(percrange.id, treenames, filenames, treesnentries, localstarts, localends,
                     globalstart, globalend, percrange.friendinfo)
