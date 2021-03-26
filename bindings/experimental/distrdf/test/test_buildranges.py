from DistRDF.Node import HeadNode
from DistRDF.Node import RangesBuilder
from DistRDF.Node import TreeInfo
import warnings
import unittest


def rangesToTuples(ranges):
    """Convert range objects to tuples with the shape (start, end)"""
    return list(map(lambda r: (r.start, r.end), ranges))


class BuildRangesTest(unittest.TestCase):
    """
    Test the building of ranges with information from the head node of the graph
    adn the RangesBuilder class.
    """

    def test_nentries_multipleOf_npartitions(self):
        """
        `BuildRanges` method when the number of entries is a multiple of the
        number of partitions.

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        nentries_small = 10
        nentries_large = 100

        # First case
        headnode.npartitions = 5
        rng = builder._get_balanced_ranges(nentries_small)
        ranges_small = rangesToTuples(rng)

        # Second case
        headnode.npartitions = 10
        rng = builder._get_balanced_ranges(nentries_large)
        ranges_large = rangesToTuples(rng)

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
        `BuildRanges` method when then number of entries is not a multiple of
        the number of partitions.

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        nentries_1 = 10
        headnode.npartitions = 4
        nentries_2 = 9

        # Example in which fractional part of
        # (nentries/npartitions) >= 0.5
        rng = builder._get_balanced_ranges(nentries_1)
        ranges_1 = rangesToTuples(rng)

        # Example in which fractional part of
        # (nentries/npartitions) < 0.5
        rng = builder._get_balanced_ranges(nentries_2)
        ranges_2 = rangesToTuples(rng)

        # Required output pairs
        ranges_1_reqd = [(0, 3), (3, 6), (6, 8), (8, 10)]
        ranges_2_reqd = [(0, 3), (3, 5), (5, 7), (7, 9)]

        self.assertListEqual(ranges_1, ranges_1_reqd)
        self.assertListEqual(ranges_2, ranges_2_reqd)

    def test_nentries_greater_than_npartitions(self):
        """
        `BuildRanges` method when the number of entries is smaller than the
        number of partitions.

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        nentries = 5
        headnode.npartitions = 7  # > nentries

        rng = builder._get_balanced_ranges(nentries)
        ranges = rangesToTuples(rng)

        ranges_reqd = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_one_cluster(self):
        """
        Check that _get_clustered_ranges returns one range when the dataset
        contains a single cluster and the number of partitions is 1

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        treename = "TotemNtuple"
        filelist = ["backend/Slimmed_ntuple.root"]
        headnode.npartitions = 1

        treeinfo = TreeInfo([treename], filelist, None, None)
        crs = builder._get_clustered_ranges(treeinfo)
        ranges = rangesToTuples(crs)

        ranges_reqd = [(0, 10)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_warning_when_npartitions_greater_than_clusters(self):
        """
        Check that _get_clustered_ranges raises a warning when the number of
        partitions is bigger than the number of clusters in the dataset.

        """

        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        treename = "TotemNtuple"
        filelist = ["backend/Slimmed_ntuple.root"]
        headnode.npartitions = 2

        ranges_reqd = [(0, 10)]
        treeinfo = TreeInfo([treename], filelist, None, None)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            crs = builder._get_clustered_ranges(treeinfo)
            ranges = rangesToTuples(crs)

            # Verify ranges
            self.assertListEqual(ranges, ranges_reqd)

            # Verify warning
            assert issubclass(w[-1].category, UserWarning)

    def test_clustered_ranges_with_two_clusters_two_partitions(self):
        """
        Check that _get_clustered_ranges creates clustered ranges respecting
        the cluster boundaries even if that implies to have ranges with very
        different numbers of entries.

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        treename = "myTree"
        filelist = ["backend/2clusters.root"]
        headnode.npartitions = 2

        treeinfo = TreeInfo([treename], filelist, None, None)
        crs = builder._get_clustered_ranges(treeinfo)
        ranges = rangesToTuples(crs)

        ranges_reqd = [
            (0, 777),
            (777, 1000)
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_four_clusters_four_partitions(self):
        """
        Check that _get_clustered_ranges creates clustered ranges as equal as
        possible for four partitions

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        treename = "myTree"
        filelist = ["backend/4clusters.root"]
        headnode.npartitions = 4

        treeinfo = TreeInfo([treename], filelist, None, None)
        crs = builder._get_clustered_ranges(treeinfo)
        ranges = rangesToTuples(crs)

        ranges_reqd = [
            (0, 250),
            (250, 500),
            (500, 750),
            (750, 1000)
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_many_clusters_four_partitions(self):
        """
        Check that _get_clustered_ranges creates clustered ranges as equal as
        possible for four partitions

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        treename = "myTree"
        filelist = ["backend/1000clusters.root"]
        headnode.npartitions = 4

        treeinfo = TreeInfo([treename], filelist, None, None)
        crs = builder._get_clustered_ranges(treeinfo)
        ranges = rangesToTuples(crs)

        ranges_reqd = [
            (0, 250),
            (250, 500),
            (500, 750),
            (750, 1000)
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_many_clusters_many_partitions(self):
        """
        Check that _get_clustered_ranges creates clustered ranges as equal as
        possible for the maximum number of possible partitions (number of
        clusters)

        """
        headnode = HeadNode(1)
        builder = RangesBuilder(headnode)

        treename = "myTree"
        filelist = ["backend/1000clusters.root"]
        headnode.npartitions = 1000

        treeinfo = TreeInfo([treename], filelist, None, None)
        crs = builder._get_clustered_ranges(treeinfo)
        ranges = rangesToTuples(crs)

        start = 0
        end = 1000
        step = 1

        ranges_reqd = [(a, b) for a, b in zip(range(start, end, step),
                                              range(step, end + 1, step))]

        self.assertListEqual(ranges, ranges_reqd)

    def test_buildranges_with_clustered_ranges(self):
        """
        Check that build_ranges produces clustered ranges when the dataset
        contains clusters.

        """
        headnode = HeadNode("myTree", "backend/1000clusters.root")
        builder = RangesBuilder(headnode)

        headnode.npartitions = 1000

        crs = builder.build_ranges()
        ranges = rangesToTuples(crs)

        start = 0
        end = 1000
        step = 1

        ranges_reqd = [(a, b) for a, b in zip(range(start, end, step),
                                              range(step, end + 1, step))]

        self.assertListEqual(ranges, ranges_reqd)

    def test_buildranges_with_balanced_ranges(self):
        """
        Check that build_ranges produces balanced ranges when there are no
        clusters involved.

        """
        headnode = HeadNode(50)
        builder = RangesBuilder(headnode)

        headnode.npartitions = 16

        crs = builder.build_ranges()
        ranges = rangesToTuples(crs)

        ranges_reqd = [
            (0, 4), (4, 8), (8, 11), (11, 14), (14, 17), (17, 20),
            (20, 23), (23, 26), (26, 29), (29, 32), (32, 35), (35, 38),
            (38, 41), (41, 44), (44, 47), (47, 50)
        ]

        self.assertListEqual(ranges, ranges_reqd)
