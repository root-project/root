import unittest
import PyRDF
from PyRDF.backend.Dist import Dist
from PyRDF.backend.Dist import FriendInfo
import ROOT
from array import array
import os


def rangesToTuples(ranges):
    """Convert range objects to tuples with the shape (start, end)"""
    return list(map(lambda r: (r.start, r.end), ranges))


class FriendInfoTest(unittest.TestCase):
    """Unit test for the FriendInfo class"""

    class TestBackend(Dist):
        """Dummy backend to test the _get_friend_info method in Dist class."""

        def ProcessAndMerge(self, mapper, reducer):
            """Dummy implementation of ProcessAndMerge."""
            pass

        def distribute_files(self, includes_list):
            """
            Dummy implementation of distribute_files. Does nothing.
            """
            pass

    def create_parent_tree(self):
        """Creates a .root file with the parent TTree"""
        f = ROOT.TFile("treeparent.root", "recreate")
        T = ROOT.TTree("T", "test friend trees")

        x = array("f", [0])
        T.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The parent will have a gaussian distribution with mean 10 and
        # standard deviation 1
        for i in range(10000):
            x[0] = r.Gaus(10, 1)
            T.Fill()

        f.Write()
        f.Close()

    def create_friend_tree(self):
        """Creates a .root file with the friend TTree"""
        ff = ROOT.TFile("treefriend.root", "recreate")
        TF = ROOT.TTree("TF", "tree friend")

        x = array("f", [0])
        TF.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The friend will have a gaussian distribution with mean 20 and
        # standard deviation 1
        for i in range(10000):
            x[0] = r.Gaus(20, 1)
            TF.Fill()

        ff.Write()
        ff.Close()

    def test_empty_friend_info(self):
        """Check that FriendInfo is initialized with two empty lists"""

        friend_info = FriendInfo()

        friend_names = friend_info.friend_names
        friend_file_names = friend_info.friend_file_names

        # Check that both lists in FriendInfo are empty
        self.assertTrue(len(friend_names) == 0)
        self.assertTrue(len(friend_file_names) == 0)

        # Check functioning of __bool__ method
        self.assertFalse(friend_info)

    def test_friend_info_with_ttree(self):
        """
        Check that FriendInfo correctly stores information about the friend
        trees
        """
        self.create_parent_tree()
        self.create_friend_tree()

        # Parent Tree
        base_tree_name = "T"
        base_tree_filename = "treeparent.root"
        baseTree = ROOT.TChain(base_tree_name)
        baseTree.Add(base_tree_filename)

        # Friend Tree
        friend_tree_name = "TF"
        friend_tree_filename = "treefriend.root"
        friendTree = ROOT.TChain(friend_tree_name)
        friendTree.Add(friend_tree_filename)

        # Add friendTree to the parent
        baseTree.AddFriend(friendTree)

        # Instantiate dummy backend
        backend = FriendInfoTest.TestBackend()

        # Retrieve FriendInfo instance
        friend_info = backend._get_friend_info(baseTree)

        # Check that FriendInfo has non-empty lists
        self.assertTrue(friend_info)

        # Check that the two lists with treenames and filenames are populated
        # as expected.
        self.assertListEqual(friend_info.friend_names, [friend_tree_name])
        self.assertListEqual(
            friend_info.friend_file_names,
            [[friend_tree_filename]]
        )

        # Remove unnecessary .root files
        os.remove(base_tree_filename)
        os.remove(friend_tree_filename)


class DistBackendInitTest(unittest.TestCase):
    """Dist abstract class cannot be instantiated."""

    def test_dist_init_error(self):
        """
        Any attempt to instantiate the `Dist` abstract class results in
        a `TypeError`.

        """
        with self.assertRaises(TypeError):
            Dist()

    def test_subclass_without_method_error(self):
        """
        Creation of a subclass without implementing `processAndMerge`
        method throws a `TypeError`.

        """
        class TestBackend(Dist):
            pass

        with self.assertRaises(TypeError):
            TestBackend()


class DistBuildRangesTest(unittest.TestCase):
    """`BuildRanges instance method in `Dist` class."""

    class TestBackend(Dist):
        """Dummy backend to test the BuildRanges method in Dist class."""

        def ProcessAndMerge(self, mapper, reducer):
            """Dummy implementation of ProcessAndMerge."""
            pass

        def distribute_files(self, includes_list):
            """
            Dummy implementation of distribute_files. Does nothing.
            """
            pass

    def test_nentries_multipleOf_npartitions(self):
        """
        `BuildRanges` method when the number of entries is a multiple of the
        number of partitions.

        """
        backend = DistBuildRangesTest.TestBackend()
        nentries_small = 10
        nentries_large = 100

        # First case
        backend.npartitions = 5
        rng = backend._get_balanced_ranges(nentries_small)
        ranges_small = rangesToTuples(rng)

        # Second case
        backend.npartitions = 10
        rng = backend._get_balanced_ranges(nentries_large)
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
        backend = DistBuildRangesTest.TestBackend()
        nentries_1 = 10
        backend.npartitions = 4
        nentries_2 = 9

        # Example in which fractional part of
        # (nentries/npartitions) >= 0.5
        rng = backend._get_balanced_ranges(nentries_1)
        ranges_1 = rangesToTuples(rng)

        # Example in which fractional part of
        # (nentries/npartitions) < 0.5
        rng = backend._get_balanced_ranges(nentries_2)
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
        backend = DistBuildRangesTest.TestBackend()
        nentries = 5
        backend.npartitions = 7  # > nentries

        rng = backend._get_balanced_ranges(nentries)
        ranges = rangesToTuples(rng)

        ranges_reqd = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_clustered_ranges_with_one_cluster(self):
        """
        Check that _get_clustered_ranges returns one range when the dataset
        contains a single cluster and the number of partitions is 1

        """
        backend = DistBuildRangesTest.TestBackend()
        treename = "TotemNtuple"
        filelist = ["Slimmed_ntuple.root"]
        backend.npartitions = 1

        crs = backend._get_clustered_ranges(treename, filelist)
        ranges = rangesToTuples(crs)

        ranges_reqd = [(0, 10)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_warning_when_npartitions_greater_than_clusters(self):
        """
        Check that _get_clustered_ranges raises a warning when the number of
        partitions is bigger than the number of clusters in the dataset.

        """
        import warnings

        backend = DistBuildRangesTest.TestBackend()
        treename = "TotemNtuple"
        filelist = ["Slimmed_ntuple.root"]
        backend.npartitions = 2

        ranges_reqd = [(0, 10)]

        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            crs = backend._get_clustered_ranges(treename, filelist)
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
        backend = DistBuildRangesTest.TestBackend()
        treename = "myTree"
        filelist = ["2clusters.root"]
        backend.npartitions = 2

        crs = backend._get_clustered_ranges(treename, filelist)
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
        backend = DistBuildRangesTest.TestBackend()
        treename = "myTree"
        filelist = ["4clusters.root"]
        backend.npartitions = 4

        crs = backend._get_clustered_ranges(treename, filelist)
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
        backend = DistBuildRangesTest.TestBackend()
        treename = "myTree"
        filelist = ["1000clusters.root"]
        backend.npartitions = 4

        crs = backend._get_clustered_ranges(treename, filelist)
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
        backend = DistBuildRangesTest.TestBackend()
        treename = "myTree"
        filelist = ["1000clusters.root"]
        backend.npartitions = 1000

        crs = backend._get_clustered_ranges(treename, filelist)
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
        backend = DistBuildRangesTest.TestBackend()

        # Mock attributes accessed through self. inside build_ranges
        backend.treename = "myTree"
        backend.files = "1000clusters.root"
        backend.friend_info = None
        backend.nentries = 1000
        backend.npartitions = 1000

        crs = backend.build_ranges()
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
        backend = DistBuildRangesTest.TestBackend()

        # Mock attributes accessed through self. inside build_ranges
        backend.treename = None
        backend.files = None
        backend.nentries = 50
        backend.npartitions = 16

        crs = backend.build_ranges()
        ranges = rangesToTuples(crs)

        ranges_reqd = [
            (0, 4), (4, 8), (8, 11), (11, 14), (14, 17), (17, 20),
            (20, 23), (23, 26), (26, 29), (29, 32), (32, 35), (35, 38),
            (38, 41), (41, 44), (44, 47), (47, 50)
        ]

        self.assertListEqual(ranges, ranges_reqd)


class DistRDataFrameInterface(unittest.TestCase):
    """
    Check `build_ranges` when instantiating RDataFrame with different
    parameters
    """

    from PyRDF import current_backend

    class TestBackend(Dist):
        """Dummy backend to test the build_ranges method in Dist class."""

        def ProcessAndMerge(self, mapper, reducer):
            """
            Dummy implementation of ProcessAndMerge.
            Return a mock list of a single value.

            """
            values = [1]
            return values

        def distribute_files(self, includes_list):
            """
            Dummy implementation of distribute_files. Does nothing.
            """
            pass

    def get_ranges_from_rdataframe(self, rdf):
        """
        Common test setup to create ranges out of an RDataFrame instance based
        on its parameters.
        """
        PyRDF.current_backend = DistRDataFrameInterface.TestBackend()
        backend = PyRDF.current_backend

        hist = rdf.Define("b1", "tdfentry_")\
                  .Histo1D("b1")

        # Trigger call to `execute` where number of entries, treename
        # and input files are extracted from the arguments passed to
        # the RDataFrame head node
        hist.GetValue()

        backend.npartitions = 2
        ranges = rangesToTuples(backend.build_ranges())
        return ranges

    def test_empty_rdataframe_with_number_of_entries(self):
        """
        An RDataFrame instantiated with a number of entries leads to balanced
        ranges.

        """
        rdf = PyRDF.RDataFrame(10)

        ranges = self.get_ranges_from_rdataframe(rdf)
        ranges_reqd = [(0, 5), (5, 10)]
        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_simple_filename(self):
        """
        Check clustered ranges produced when the input dataset is a single ROOT
        file.

        """
        treename = "myTree"
        filename = "2clusters.root"
        rdf = PyRDF.RDataFrame(treename, filename)

        ranges = self.get_ranges_from_rdataframe(rdf)
        ranges_reqd = [(0, 777), (777, 1000)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_filename_with_globbing(self):
        """
        Check clustered ranges produced when the input dataset is a single ROOT
        file with globbing.

        """
        treename = "myTree"
        filename = "2cluste*.root"
        rdf = PyRDF.RDataFrame(treename, filename)

        ranges = self.get_ranges_from_rdataframe(rdf)
        ranges_reqd = [(0, 777), (777, 1000)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_list_of_one_file(self):
        """
        Check clustered ranges produced when the input dataset is a list of a
        single ROOT file.

        """
        treename = "myTree"
        filelist = ["2clusters.root"]
        rdf = PyRDF.RDataFrame(treename, filelist)

        ranges = self.get_ranges_from_rdataframe(rdf)
        ranges_reqd = [(0, 777), (777, 1000)]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_list_of_files(self):
        """
        Check clustered ranges produced when the dataset is a list of a multiple
        ROOT files.

        Explanation about required ranges:
        - 2clusters.root contains 1000 entries split into 2 clusters
            ([0, 776], [777, 999]) being 776 and 999 inclusive entries
        - 4clusters.root contains 1000 entries split into 4 clusters
            ([0, 249], [250, 499], [500, 749], [750, 999]) being 249, 499, 749
            and 999 inclusive entries

        Current mechanism to create clustered ranges takes only into account the
        the number of clusters, it is assumed that clusters inside a ROOT file
        are properly distributed and balanced with respect to the number of
        entries.

        Thus, if a dataset is composed by two ROOT files which are poorly
        balanced in terms of clusters and entries, the resultant ranges will
        still respect the cluster boundaries but each one may contain a
        different number of entries.

        Since this case should not be common, ranges required on this test are
        considered the expected result.
        """
        treename = "myTree"
        filelist = ["2clusters.root",
                    "4clusters.root"]

        rdf = PyRDF.RDataFrame(treename, filelist)

        ranges = self.get_ranges_from_rdataframe(rdf)
        ranges_reqd = [(0, 1250), (250, 1000)]

        self.assertListEqual(ranges, ranges_reqd)
