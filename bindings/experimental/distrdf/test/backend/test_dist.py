import os
import unittest

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF import Proxy
from DistRDF.Backends import Base

import ROOT


def emptysourceranges_to_tuples(ranges):
    """Convert EmptySourceRange objects to tuples with the shape (start, end)"""
    return [(r.start, r.end) for r in ranges]


def treeranges_to_tuples(ranges):
    """Convert TreeRange objects to tuples with the shape (start, end, filelist)"""
    return [(r.globalstart, r.globalend, r.filenames, r.treenames) for r in ranges]


class BaseBackendInitTest(unittest.TestCase):
    """Dist abstract class cannot be instantiated."""

    def test_dist_init_error(self):
        """
        Any attempt to instantiate the `Dist` abstract class results in
        a `TypeError`.

        """
        with self.assertRaises(TypeError):
            Base.BaseBackend()

    def test_subclass_without_method_error(self):
        """
        Creation of a subclass without implementing `processAndMerge`
        method throws a `TypeError`.

        """
        class TestBackend(Base.BaseBackend):
            pass

        with self.assertRaises(TypeError):
            TestBackend()


class DistRDataFrameInterface(unittest.TestCase):
    """
    Check `build_ranges` when instantiating RDataFrame with different
    parameters
    """

    class TestBackend(Base.BaseBackend):
        """Dummy backend to test the build_ranges method in Dist class."""

        def ProcessAndMerge(self, ranges, mapper, reducer):
            """
            Dummy implementation of ProcessAndMerge.
            Return a mock list of a single value.

            """

            class DummyValue(object):
                """
                Dummy value class to avoid triggering Spark execution. Needed
                to provide a `GetValue` method to call in `Dist.execute`.
                """

                def GetValue(self):
                    return 1

            return [DummyValue()]

        def distribute_unique_paths(self, includes_list):
            """
            Dummy implementation of distribute_files. Does nothing.
            """
            pass

        def make_dataframe(self, *args, **kwargs):
            """Dummy make_dataframe"""
            pass

    class TestDataFrame(object):
        """
        Interface to an RDataFrame that can run its computation graph
        distributedly.
        """

        def __init__(self, *args):
            """initialize"""

            # Passing None as `npartitions`, the tests will change it as needed.
            self.headnode = HeadNode.get_headnode(None, *args)

            self.headnode.backend = DistRDataFrameInterface.TestBackend()

            self.headproxy = Proxy.TransformationProxy(self.headnode)

        def __getattr__(self, attr):
            """getattr"""
            return getattr(self.headproxy, attr)

    def get_ranges_from_empty_rdataframe(self, rdf):
        """
        Common test setup to create ranges out of an empty source based
        RDataFrame instance.
        """
        headnode = rdf.headnode
        headnode.npartitions = 2

        hist = rdf.Define("b1", "tdfentry_")\
                  .Histo1D("b1")

        # Trigger call to `execute` where number of entries, treename
        # and input files are extracted from the arguments passed to
        # the RDataFrame head node
        hist.GetValue()

        ranges = emptysourceranges_to_tuples(headnode.build_ranges())
        return ranges

    def get_ranges_from_tree_rdataframe(self, rdf):
        """
        Common test setup to create ranges out of an TTree based RDataFrame
        instance.
        """
        headnode = rdf.headnode
        headnode.npartitions = 2

        hist = rdf.Define("b1", "tdfentry_")\
                  .Histo1D("b1")

        # Trigger call to `execute` where number of entries, treename
        # and input files are extracted from the arguments passed to
        # the RDataFrame head node
        hist.GetValue()

        ranges = treeranges_to_tuples(headnode.build_ranges())
        return ranges

    def test_empty_rdataframe_with_number_of_entries(self):
        """
        An RDataFrame instantiated with a number of entries leads to balanced
        ranges.

        """
        rdf = DistRDataFrameInterface.TestDataFrame(10)

        ranges = self.get_ranges_from_empty_rdataframe(rdf)
        ranges_reqd = [(0, 5), (5, 10)]
        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_simple_filename(self):
        """
        Check clustered ranges produced when the input dataset is a single ROOT
        file.

        """
        treename = "myTree"
        filename = "2clusters.root"
        rdf = DistRDataFrameInterface.TestDataFrame(treename, filename)

        ranges = self.get_ranges_from_tree_rdataframe(rdf)
        ranges_reqd = [(0, 777, [filename], [treename]),
                       (777, 1000, [filename], [treename])]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_filename_with_globbing(self):
        """
        Check clustered ranges produced by the distributed RDataFrame when the
        input dataset is glob that expands to a single ROOT file in the same
        folder.

        """
        treename = "myTree"
        filename = "2cluste*.root"
        rdf = DistRDataFrameInterface.TestDataFrame(treename, filename)

        # Need absolute path because TChain globbing expands to full paths
        expected_inputfiles = [os.path.join(os.getcwd(), "2clusters.root")]
        ranges = self.get_ranges_from_tree_rdataframe(rdf)
        ranges_reqd = [(0, 777,  expected_inputfiles, [treename]),
                       (777, 1000,  expected_inputfiles, [treename])]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_treename_and_list_of_one_file(self):
        """
        Check clustered ranges produced when the input dataset is a list of a
        single ROOT file.

        """
        treename = "myTree"
        filelist = ["2clusters.root"]
        rdf = DistRDataFrameInterface.TestDataFrame(treename, filelist)

        ranges = self.get_ranges_from_tree_rdataframe(rdf)
        ranges_reqd = [(0, 777,  filelist, [treename]),
                       (777, 1000, filelist, [treename])]

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

        rdf = DistRDataFrameInterface.TestDataFrame(treename, filelist)

        ranges = self.get_ranges_from_tree_rdataframe(rdf)
        ranges_reqd = [
            (0, 1250, ["2clusters.root", "4clusters.root"], [treename, treename]),
            (250, 1000, ["4clusters.root"], [treename]),
        ]

        self.assertListEqual(ranges, ranges_reqd)

    def test_rdataframe_with_notreename_and_chain_with_subtrees(self):
        """
        Checks that we retrieve all the information to process a TChain with
        sub trees with different names in a distributed task.
        """
        # Create two dummy files
        treename1, filename1 = "entries_1", "entries_1.root"
        treename2, filename2 = "entries_2", "entries_2.root"
        ROOT.RDataFrame(10).Define("x", "rdfentry_").Snapshot(treename1, filename1)
        ROOT.RDataFrame(10).Define("x", "rdfentry_").Snapshot(treename2, filename2)

        chain = ROOT.TChain()
        chain.Add(str(filename1 + "/" + treename1))
        chain.Add(str(filename2 + "/" + treename2))

        rdf = DistRDataFrameInterface.TestDataFrame(chain)
        ranges = self.get_ranges_from_tree_rdataframe(rdf)
        ranges_reqd = [
            (0, 10, [filename1], [treename1]),
            (0, 10, [filename2], [treename2])
        ]

        os.remove(filename1)
        os.remove(filename2)
        self.assertListEqual(ranges, ranges_reqd)


class DistRDataFrameInvariants(unittest.TestCase):
    """
    The result of distributed execution should not depend on the number of
    partitions assigned to the dataframe.
    """

    class TestBackend(Base.BaseBackend):
        """Dummy backend to test the build_ranges method in Dist class."""

        def ProcessAndMerge(self, ranges, mapper, reducer):
            """
            Dummy implementation of ProcessAndMerge.
            """
            mergeables_lists = [mapper(range) for range in ranges]

            while len(mergeables_lists) > 1:
                mergeables_lists.append(
                    reducer(mergeables_lists.pop(0), mergeables_lists.pop(0)))

            return mergeables_lists.pop()

        def distribute_unique_paths(self, includes_list):
            """
            Dummy implementation of distribute_unique_paths. Does nothing.
            """
            pass

        def make_dataframe(self, *args, **kwargs):
            """Dummy make_dataframe"""
            pass

    def test_count_result_invariance(self):
        """
        Tests that counting the entries in the dataset does not depend on the
        number of partitions. This could have happened if we used TEntryList
        to restrict processing on a certain range of entries of the TChain in a
        distributed task, but the changes in
        https://github.com/root-project/root/commit/77bd5aa82e9544811e0d5fce197ab87c739c2e23
        were not implemented yet.
        """
        treename = "entries"
        filenames = ["1cluster_20entries.root"] * 5

        for npartitions in range(1, 6):
            headnode = HeadNode.get_headnode(npartitions, treename, filenames)
            backend = DistRDataFrameInvariants.TestBackend()
            rdf = DataFrame.RDataFrame(headnode, backend)
            self.assertEqual(rdf.Count().GetValue(), 100)
