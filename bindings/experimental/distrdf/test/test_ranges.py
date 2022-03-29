import os
import unittest

from DistRDF.HeadNode import get_headnode
from DistRDF import Ranges

import ROOT


def emptysourceranges_to_tuples(ranges):
    """Convert EmptySourceRange objects to tuples with the shape (start, end)"""
    return [(r.start, r.end) for r in ranges]


def treeranges_to_tuples(ranges):
    """Convert TreeRange objects to tuples with the shape (start, end, filenames)."""
    return [(r.globalstart, r.globalend, r.localstarts, r.localends, r.filenames) for r in ranges]


class EmptySourceRanges(unittest.TestCase):
    """
    Test cases with ranges when there is an empty data source.
    """

    def test_nentries_multipleOf_npartitions(self):
        """
        Building balanced ranges when the number of entries is a multiple of the
        number of partitions.
        """

        nentries_small = 10
        npartitions_small = 5
        nentries_large = 100
        npartitions_large = 10

        # First case
        rng = Ranges.get_balanced_ranges(nentries_small, npartitions_small)
        ranges_small = emptysourceranges_to_tuples(rng)

        # Second case
        rng = Ranges.get_balanced_ranges(nentries_large, npartitions_large)
        ranges_large = emptysourceranges_to_tuples(rng)

        ranges_small_reqd = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        ranges_large_reqd = [
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 70),
            (70, 80),
            (80, 90),
            (90, 100)
        ]

        self.assertListEqual(ranges_small, ranges_small_reqd)
        self.assertListEqual(ranges_large, ranges_large_reqd)

    def test_nentries_not_multipleOf_npartitions(self):
        """
        Building balanced ranges when the number of entries is not a multiple of
        the number of partitions.
        """

        nentries_1 = 10
        nentries_2 = 9
        npartitions = 4

        # Example in which fractional part of
        # (nentries/npartitions) >= 0.5
        rng = Ranges.get_balanced_ranges(nentries_1, npartitions)
        ranges_1 = emptysourceranges_to_tuples(rng)

        # Example in which fractional part of
        # (nentries/npartitions) < 0.5
        rng = Ranges.get_balanced_ranges(nentries_2, npartitions)
        ranges_2 = emptysourceranges_to_tuples(rng)

        # Required output pairs
        ranges_1_reqd = [(0, 3), (3, 6), (6, 8), (8, 10)]
        ranges_2_reqd = [(0, 3), (3, 5), (5, 7), (7, 9)]

        self.assertListEqual(ranges_1, ranges_1_reqd)
        self.assertListEqual(ranges_2, ranges_2_reqd)

    def test_nentries_greater_than_npartitions(self):
        """
        Building balanced ranges when the number of entries is smaller than the
        number of partitions.
        """

        nentries = 5
        npartitions = 7

        rng = Ranges.get_balanced_ranges(nentries, npartitions)
        ranges = emptysourceranges_to_tuples(rng)

        ranges_reqd = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_buildranges_with_balanced_ranges(self):
        """
        Check that build_ranges produces balanced ranges when there are no
        clusters involved.
        """
        npartitions = 16
        nentries = 50
        headnode = get_headnode(npartitions, nentries)

        crs = headnode.build_ranges()
        ranges = emptysourceranges_to_tuples(crs)

        ranges_reqd = [
            (0, 4), (4, 8), (8, 11), (11, 14), (14, 17), (17, 20),
            (20, 23), (23, 26), (26, 29), (29, 32), (32, 35), (35, 38),
            (38, 41), (41, 44), (44, 47), (47, 50)
        ]

        self.assertListEqual(ranges, ranges_reqd)


class TreeRanges(unittest.TestCase):
    """
    Test cases with ranges when the data source is a TTree.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create some files to be used in the tests. Each file has 100 entries and
        10 clusters.
        """

        opts = ROOT.RDF.RSnapshotOptions()
        opts.fAutoFlush = 10
        df = ROOT.RDataFrame(100).Define("x", "1")
        for i in range(3):
            df.Snapshot(f"tree_{i}", f"distrdf_unittests_file_{i}.root", ["x"], opts)

    @classmethod
    def tearDownClass(cls):
        """Destroy the previously created files."""
        for i in range(3):
            os.remove(f"distrdf_unittests_file_{i}.root")

    def test_clustered_ranges_with_one_cluster(self):
        """
        Exactly one range is created when user asks for one partition. The range
        spans the whole input file.
        """

        # This tree has 10 entries and 1 cluster
        treenames = ["TotemNtuple"]
        filenames = ["backend/Slimmed_ntuple.root"]
        npartitions = 1

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)

        self.assertEqual(len(ranges), npartitions)

        ranges_reqd = [(0, 10, [0], [10], ["backend/Slimmed_ntuple.root"])]

        self.assertListEqual(ranges, ranges_reqd)

    def test_npartitions_greater_than_clusters(self):
        """
        Asking for 2 partitions with an input file that contains only 1 cluster
        returns a list with two tasks. One spans the whole file, the other is
        None.
        """

        # This tree has 10 entries and 1 cluster
        treenames = ["TotemNtuple"]
        filenames = ["backend/Slimmed_ntuple.root"]
        npartitions = 2

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        # We return one task per partition
        self.assertEqual(len(clusteredranges), npartitions)
        # But only one is non-empty
        actualtasks = [task for task in clusteredranges if task is not None]
        self.assertEqual(len(actualtasks), 1)

        ranges = treeranges_to_tuples(actualtasks)
        ranges_reqd = [(0, 10, [0], [10], ["backend/Slimmed_ntuple.root"])]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_two_clusters_two_partitions(self):
        """
        Create clustered ranges respecting the cluster boundaries, even if that
        implies to have ranges with different numbers of entries.
        """

        treenames = ["myTree"]
        filenames = ["backend/2clusters.root"]
        npartitions = 2

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]
        ranges = treeranges_to_tuples(clusteredranges)

        ranges_reqd = [
            (0, 777, [0], [777], ["backend/2clusters.root"]),
            (777, 1000, [777], [1000], ["backend/2clusters.root"])
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_filename_with_globbing(self):
        """
        Check globbing returns the proper file name to create ranges.
        """
        treename = "myTree"
        filename = "backend/2cluste*.root"
        npartitions = 2
        rdf = get_headnode(npartitions, treename, filename)

        expected_inputfiles = ["backend/2clusters.root"]
        extracted_inputfiles = rdf.inputfiles

        percranges = Ranges.get_percentage_ranges([treename], extracted_inputfiles, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]
        ranges = treeranges_to_tuples(clusteredranges)

        ranges_reqd = [
            (0, 777, [0], [777], expected_inputfiles),
            (777, 1000, [777], [1000], expected_inputfiles)
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_notreename_and_chain_with_subtrees(self):
        """
        Check proper handling of a TChain with different subnames.
        """
        # Create two dummy files
        treename1, filename1 = "entries_1", "entries_1.root"
        treename2, filename2 = "entries_2", "entries_2.root"
        npartitions = 2
        ROOT.RDataFrame(10).Define("x", "rdfentry_").Snapshot(treename1, filename1)
        ROOT.RDataFrame(10).Define("x", "rdfentry_").Snapshot(treename2, filename2)

        chain = ROOT.TChain()
        chain.Add(str(filename1 + "?#" + treename1))
        chain.Add(str(filename2 + "?#" + treename2))

        rdf = get_headnode(npartitions, chain)
        extracted_subtreenames = rdf.subtreenames
        extracted_filenames = rdf.inputfiles

        percranges = Ranges.get_percentage_ranges(
            extracted_subtreenames, extracted_filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]
        ranges = treeranges_to_tuples(clusteredranges)

        ranges_reqd = [
            (0, 10, [0], [10], [filename1]),
            (0, 10, [0], [10], [filename2])
        ]

        os.remove(filename1)
        os.remove(filename2)
        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_four_clusters_four_partitions(self):
        """
        When the cluster boundaries allow it, create ranges as equal as possible
        in terms of how many entries they span.
        """

        treenames = ["myTree"]
        filenames = ["backend/4clusters.root"]
        npartitions = 4

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)

        ranges_reqd = [
            (0, 250, [0], [250], ["backend/4clusters.root"]),
            (250, 500, [250], [500], ["backend/4clusters.root"]),
            (500, 750, [500], [750], ["backend/4clusters.root"]),
            (750, 1000, [750], [1000], ["backend/4clusters.root"])
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_many_clusters_four_partitions(self):
        """
        Create ranges that spany many clusters.
        """

        treenames = ["myTree"]
        filenames = ["backend/1000clusters.root"]
        npartitions = 4

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)

        ranges_reqd = [
            (0, 250, [0], [250], ["backend/1000clusters.root"]),
            (250, 500, [250], [500], ["backend/1000clusters.root"]),
            (500, 750, [500], [750], ["backend/1000clusters.root"]),
            (750, 1000, [750], [1000], ["backend/1000clusters.root"])
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_many_clusters_many_partitions(self):
        """
        Create as many partitions as number of clusters in the file.
        """

        treenames = ["myTree"]
        filenames = ["backend/1000clusters.root"]
        npartitions = 1000

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)

        start = 0
        end = 1000
        step = 1

        ranges_reqd = [(a, b, [a], [b], filenames) for a, b
                       in zip(range(start, end, step), range(step, end + 1, step))]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_two_files(self):
        """
        Create two ranges from two files with a different number of clusters.
        """
        treenames = ["myTree"] * 2
        filenames = ["backend/2clusters.root", "backend/4clusters.root"]
        npartitions = 2

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)
        ranges_reqd = [(0, 1000, [0], [1000], [filename]) for filename in filenames]

        self.assertListEqual(ranges, ranges_reqd)

    def test_three_files_one_partition(self):
        """
        Create one range that spans three files.
        """
        nfiles = 3
        treenames = [f"tree_{i}" for i in range(nfiles)]
        filenames = [f"distrdf_unittests_file_{i}.root" for i in range(nfiles)]
        npartitions = 1

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)
        ranges_reqd = [(0, 300, [0, 0, 0], [100, 100, 100], filenames)]
        self.assertListEqual(ranges, ranges_reqd)

    def test_three_files_one_partition_per_file(self):
        """
        Create as many ranges as files
        """
        nfiles = 3
        treenames = [f"tree_{i}" for i in range(nfiles)]
        filenames = [f"distrdf_unittests_file_{i}.root" for i in range(nfiles)]
        npartitions = nfiles

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)
        ranges_reqd = [(0, 100, [0], [100], [filename]) for filename in filenames]

        self.assertListEqual(ranges, ranges_reqd)

    def test_three_files_two_partitions_per_file(self):
        """
        Create two partitions per file
        """
        nfiles = 3
        treenames = [f"tree_{i}" for i in range(nfiles)]
        filenames = [f"distrdf_unittests_file_{i}.root" for i in range(nfiles)]
        npartitions = nfiles * 2
        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)
        ranges_reqd = [
            # File 0
            (0, 50, [0], [50], [filenames[0]]),
            (50, 100, [50], [100], [filenames[0]]),
            # File 1
            (0, 50, [0], [50], [filenames[1]]),
            (50, 100, [50], [100], [filenames[1]]),
            # File 2
            (0, 50, [0], [50], [filenames[2]]),
            (50, 100, [50], [100], [filenames[2]]),
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_three_files_partitions_equal_clusters(self):
        """
        Create as many partitions as clusters in the dataset.
        """
        nfiles = 3
        treenames = [f"tree_{i}" for i in range(nfiles)]
        filenames = [f"distrdf_unittests_file_{i}.root" for i in range(nfiles)]
        npartitions = nfiles * 10  # trees have 10 clusters

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        ranges = treeranges_to_tuples(clusteredranges)
        ranges_reqd = [
            (0, 10, [0], [10], [filenames[0]]),
            (10, 20, [10], [20], [filenames[0]]),
            (20, 30, [20], [30], [filenames[0]]),
            (30, 40, [30], [40], [filenames[0]]),
            (40, 50, [40], [50], [filenames[0]]),
            (50, 60, [50], [60], [filenames[0]]),
            (60, 70, [60], [70], [filenames[0]]),
            (70, 80, [70], [80], [filenames[0]]),
            (80, 90, [80], [90], [filenames[0]]),
            (90, 100, [90], [100], [filenames[0]]),
            (0, 10, [0], [10], [filenames[1]]),
            (10, 20, [10], [20], [filenames[1]]),
            (20, 30, [20], [30], [filenames[1]]),
            (30, 40, [30], [40], [filenames[1]]),
            (40, 50, [40], [50], [filenames[1]]),
            (50, 60, [50], [60], [filenames[1]]),
            (60, 70, [60], [70], [filenames[1]]),
            (70, 80, [70], [80], [filenames[1]]),
            (80, 90, [80], [90], [filenames[1]]),
            (90, 100, [90], [100], [filenames[1]]),
            (0, 10, [0], [10], [filenames[2]]),
            (10, 20, [10], [20], [filenames[2]]),
            (20, 30, [20], [30], [filenames[2]]),
            (30, 40, [30], [40], [filenames[2]]),
            (40, 50, [40], [50], [filenames[2]]),
            (50, 60, [50], [60], [filenames[2]]),
            (60, 70, [60], [70], [filenames[2]]),
            (70, 80, [70], [80], [filenames[2]]),
            (80, 90, [80], [90], [filenames[2]]),
            (90, 100, [90], [100], [filenames[2]])
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_three_files_partitions_greater_than_clusters(self):
        """
        Create more partitions than clusters in the dataset.
        """
        nfiles = 3
        treenames = [f"tree_{i}" for i in range(nfiles)]
        filenames = [f"distrdf_unittests_file_{i}.root" for i in range(nfiles)]
        npartitions = 42

        percranges = Ranges.get_percentage_ranges(treenames, filenames, npartitions, friendinfo=None)
        clusteredranges = [Ranges.get_clustered_range_from_percs(percrange) for percrange in percranges]

        # We return one task per partition
        self.assertEqual(len(clusteredranges), npartitions)
        # But at most as many as the number of clusters in the dataset are non-empty
        actualtasks = [task for task in clusteredranges if task is not None]
        self.assertEqual(len(actualtasks), 30)

        # Same as previous test
        ranges = treeranges_to_tuples(actualtasks)
        ranges_reqd = [
            (0, 10, [0], [10], [filenames[0]]),
            (10, 20, [10], [20], [filenames[0]]),
            (20, 30, [20], [30], [filenames[0]]),
            (30, 40, [30], [40], [filenames[0]]),
            (40, 50, [40], [50], [filenames[0]]),
            (50, 60, [50], [60], [filenames[0]]),
            (60, 70, [60], [70], [filenames[0]]),
            (70, 80, [70], [80], [filenames[0]]),
            (80, 90, [80], [90], [filenames[0]]),
            (90, 100, [90], [100], [filenames[0]]),
            (0, 10, [0], [10], [filenames[1]]),
            (10, 20, [10], [20], [filenames[1]]),
            (20, 30, [20], [30], [filenames[1]]),
            (30, 40, [30], [40], [filenames[1]]),
            (40, 50, [40], [50], [filenames[1]]),
            (50, 60, [50], [60], [filenames[1]]),
            (60, 70, [60], [70], [filenames[1]]),
            (70, 80, [70], [80], [filenames[1]]),
            (80, 90, [80], [90], [filenames[1]]),
            (90, 100, [90], [100], [filenames[1]]),
            (0, 10, [0], [10], [filenames[2]]),
            (10, 20, [10], [20], [filenames[2]]),
            (20, 30, [20], [30], [filenames[2]]),
            (30, 40, [30], [40], [filenames[2]]),
            (40, 50, [40], [50], [filenames[2]]),
            (50, 60, [50], [60], [filenames[2]]),
            (60, 70, [60], [70], [filenames[2]]),
            (70, 80, [70], [80], [filenames[2]]),
            (80, 90, [80], [90], [filenames[2]]),
            (90, 100, [90], [100], [filenames[2]])
        ]

        self.assertListEqual(ranges, ranges_reqd)
