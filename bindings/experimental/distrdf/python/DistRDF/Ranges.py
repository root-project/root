import logging

from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import accumulate
from math import floor

from typing import Dict, List, Optional, Tuple

import ROOT

logger = logging.getLogger(__name__)


class DataRange:
    """
    A logical range of entries in which a dataset is split. Depending on the
    input data source, this can have different attributes.
    """
    pass


class EmptySourceRange(DataRange):
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


@dataclass
class TreeRange(DataRange):
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
    globalstart: int
    globalend: int
    friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo]


@dataclass
class TreeRangePerc(DataRange):
    """
    Range of percentages to be considered for a list of trees. Building block
    for an actual range of entries of a distributed task.
    """
    id: int
    treenames: List[str]
    filenames: List[str]
    first_file_idx: int
    last_file_idx: int
    first_tree_start_perc: float
    last_tree_end_perc: float
    friendinfo: Optional[ROOT.Internal.TreeUtils.RFriendInfo]


@dataclass
class TaskTreeEntries:
    """
    Entries corresponding to each tree assigned to a certain task, plus the
    actual number of entries that task will be processing. This information will
    be aggregated along with the main mergeable results in distributed
    execution. It serves as a sanity check that exactly the total amount of
    entries in the dataset is processed in the application.
    """
    processed_entries: int = 0
    trees_with_entries: Dict[str, int] = field(default_factory=dict)


def get_clusters_and_entries(treename: str, filename: str) -> Tuple[List[int], int]:
    """
    Retrieve cluster boundaries and number of entries of a TTree.
    """

    with ROOT.TFile.Open(filename, "READ_WITHOUT_GLOBALREGISTRATION") as tfile:
        ttree = tfile.Get(treename)

        entries: int = ttree.GetEntriesFast()

        it = ttree.GetClusterIterator(0)
        cluster_startentry: int = it()
        clusters: List[int] = [cluster_startentry]

        while cluster_startentry < entries:
            cluster_startentry = it()
            clusters.append(cluster_startentry)

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

    # Compute the starting percentage of the first tree and the ending percentage
    # of the last tree in each task.
    first_tree_start_perc_tasks = percentages_wrt_files[:-1]
    # When computing the ending percentages, if the percentage defined above is
    # zero, i.e. we are at file boundary, we want to consider the whole tree,
    # thus we set it to one.
    last_tree_end_perc_tasks = [perc if perc > 0 else 1 for perc in percentages_wrt_files[1:]]

    if friendinfo is not None:
        # We need to transmit the full list of treenames and filenames to each
        # task, in order to properly align the full dataset considering friends.
        return [
            TreeRangePerc(rangeid, treenames, filenames, start_sample_idxs[rangeid], end_sample_idxs[rangeid],
                          first_tree_start_perc_tasks[rangeid], last_tree_end_perc_tasks[rangeid], friendinfo)
            for rangeid in range(npartitions)
        ]
    else:
        # With the indexes created above, we can partition the lists of names of
        # files and trees. Each task will get a number of trees dictated by the
        # starting index (inclusive) and the ending index (exclusive) computed
        # from the full list of filenames.
        tasktreenames = [treenames[s:e] for s, e in zip(start_sample_idxs, end_sample_idxs)]
        taskfilenames = [filenames[s:e] for s, e in zip(start_sample_idxs, end_sample_idxs)]
        # On the other hand, when creating the TreeRangePerc tasks below, the
        # starting and ending indexes have to be task-local. In practice, the
        # task always starts from file index 0 and it always ends at file index
        # equal to the number of files assigned to that task.
        return [
            TreeRangePerc(
                rangeid, tasktreenames[rangeid], taskfilenames[rangeid], 0, len(taskfilenames[rangeid]),
                first_tree_start_perc_tasks[rangeid], last_tree_end_perc_tasks[rangeid], friendinfo
            )
            for rangeid in range(npartitions)
        ]


def get_entryrange_at_cluster_boundaries(percstart: float, percend: float,
                                         entries: int, clusters: List[int]) -> Tuple[int, int, int]:
    """
    Computes the pair (start, end) entries of this tree, aligned at cluster
    boundaries.

    Args:
        percstart: Starting percentage of the tree to be considered.

        percend: Ending percentage of the tree to be considered.

        entries: Total entries in this tree.

        clusters: List of cluster boundaries in this tree. It is important that
        both the start entry of the first cluster (0) as well as the end entry
        of the last cluster (i.e. entries) are included in the list. For example,
        in a tree with 100 entries and 10 clusters it could look like this:
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    Returns:
        A tuple of three elements:
            1. The starting entry in this tree aligned at the corresponding
               cluster boundary.
            2. The ending entry in this tree aligned at the corresponding
               cluster boundary.
            3. A number representing the entries that should be discarded after
               computing the clusters corresponding to the input percentages.
               This number is either 0 or the number of entries in the tree if
               the computed start and end cluster indexes are the same.
    """

    startentry = int(percstart * entries)
    endentry = int(percend * entries)
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
    entries_to_discard = entries if startcluster == endcluster else 0

    tree_startentry_at_cluster_boundary = clusters[startcluster]
    tree_endentry_at_cluster_boundary = clusters[endcluster]

    return tree_startentry_at_cluster_boundary, tree_endentry_at_cluster_boundary, entries_to_discard


def get_clustered_range_from_percs(percrange: TreeRangePerc) -> Tuple[Optional[TreeRange], TaskTreeEntries]:
    """
    Builds a range of entries to be processed for each tree in the chain created
    in a distributed task.
    """

    # first file index is inclusive
    first_file_idx = percrange.first_file_idx
    # last file index is exclusive
    last_file_idx = percrange.last_file_idx

    # Retrieve information from the trees assigned to this task. In case there
    # are friends, all files in the dataset are opened and their number of
    # entries are retrieved in order to ensure friend alignment.
    all_clusters_entries = (
        get_clusters_and_entries(treename, filename)
        for treename, filename in zip(percrange.treenames, percrange.filenames)
    )
    all_clusters, all_entries = zip(*all_clusters_entries)
    # Computing the offset of each tree is a cumulative sum over the entries in
    # the dataset. The initial offset is zero, so we define it in the following
    # tuple, since the 'accumulate' function does not accept an 'initial'
    # keyword argument until Python 3.8. In general, the offsets should be
    # lagging behind the list of tree entries by one, like so:
    # entries = [10, 10, 10, 10]
    # offsets = [0, 10, 20, 30]
    initial = (0,)
    all_offsets = tuple(accumulate(initial+all_entries[:-1]))

    # Connect each tree in each file with its number of entries
    trees_with_entries: Dict[str, int] = {
        filename + "?#" + treename: entries
        for filename, treename, entries
        in zip(
            percrange.filenames[first_file_idx:last_file_idx],
            percrange.treenames[first_file_idx:last_file_idx],
            all_entries[first_file_idx:last_file_idx]
        )
    }

    # Compute the pair (start, end) entries aligned to cluster boundaries of the
    # first and last tree in this task. These will be used to compute the
    # globalstart and globalend values. If the computed starting cluster is
    # equal to the ending cluster of a particular tree, it means that tree
    # should not be processed.
    if (last_file_idx - first_file_idx) == 1:
        # Compute only once if there is only one file
        first_tree_startentry, last_tree_endentry, first_tree_entries_to_discard = get_entryrange_at_cluster_boundaries(
            percrange.first_tree_start_perc, percrange.last_tree_end_perc,
            all_entries[first_file_idx], all_clusters[first_file_idx]
        )
        entries_to_discard = first_tree_entries_to_discard
    else:
        first_tree_startentry, _, first_tree_entries_to_discard = get_entryrange_at_cluster_boundaries(
            percrange.first_tree_start_perc, 1, all_entries[first_file_idx], all_clusters[first_file_idx]
        )
        _, last_tree_endentry, last_tree_entries_to_discard = get_entryrange_at_cluster_boundaries(
            0, percrange.last_tree_end_perc, all_entries[last_file_idx-1], all_clusters[last_file_idx-1]
        )
        entries_to_discard = first_tree_entries_to_discard + last_tree_entries_to_discard

    # The number of entries of the trees that actually contribute to this task.
    # It is equal to the sum of all entries of the trees that were assigned,
    # eventually minus the entries of the trees that should be discarded
    # according to the computation above.
    total_entries_in_task_chain = sum(all_entries[first_file_idx:last_file_idx]) - entries_to_discard

    if total_entries_in_task_chain == 0:
        # This is an empty task. This can happen:
        # - If all trees assigned to this task are empty.
        # - If the computed starting cluster is equal to the ending cluster.
        # These would effectively lead to creating a TChain with zero usable
        # entries.
        return None, TaskTreeEntries(0, trees_with_entries)

    globalstart = all_offsets[first_file_idx] + first_tree_startentry
    globalend = all_offsets[last_file_idx-1] + last_tree_endentry

    entries_in_trees = TaskTreeEntries(globalend - globalstart, trees_with_entries)

    treerange = TreeRange(percrange.id, percrange.treenames, percrange.filenames,
                            globalstart, globalend, percrange.friendinfo)

    return treerange, entries_in_trees
