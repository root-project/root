import os
import unittest

from ROOT._distrdf import DataFrame, HeadNode
from ROOT._distrdf.Backends import Base


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


class DistRDataFrameInvariants(unittest.TestCase):
    """
    The result of distributed execution should not depend on the number of
    partitions assigned to the dataframe.
    """

    class TestBackend(Base.BaseBackend):
        """Dummy backend to test the _build_ranges method in Dist class."""

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
            return args[0]
    
        def optimize_npartitions(self):
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
            backend = DistRDataFrameInvariants.TestBackend()
            headnode = HeadNode.get_headnode(backend, npartitions, treename, filenames)
            rdf = DataFrame.RDataFrame(headnode)
            self.assertEqual(rdf.Count().GetValue(), 100)

    def test_snapshot_return_filenames(self):
        """Test Snapshot with return_filenames=True returns list of partial files."""
        treename = "entries"
        filenames = ["1cluster_20entries.root"] * 2
        backend = DistRDataFrameInvariants.TestBackend()
        headnode = HeadNode.get_headnode(backend, 2, treename, filenames)
        rdf = DataFrame.RDataFrame(headnode)
        output_file = "test_snapshot_return_filenames_output.root"
        try:
            files = rdf.Snapshot(treename, output_file, return_filenames=True)
            self.assertIsInstance(files, list)
            self.assertEqual(len(files), 2)
            for f in files:
                self.assertTrue(os.path.exists(f))
        finally:
            for i in range(2):
                p = f"test_snapshot_return_filenames_output_{i}.root"
                if os.path.exists(p):
                    os.remove(p)

    def test_snapshot_merge(self):
        """Test Snapshot with merge=True merges partial files into target file."""
        treename = "entries"
        filenames = ["1cluster_20entries.root"] * 2
        backend = DistRDataFrameInvariants.TestBackend()
        headnode = HeadNode.get_headnode(backend, 2, treename, filenames)
        rdf = DataFrame.RDataFrame(headnode)
        output_file = "test_snapshot_merge_output.root"
        try:
            rdf.Snapshot(treename, output_file, merge=True)
            self.assertTrue(os.path.exists(output_file))
            for i in range(2):
                self.assertFalse(os.path.exists(f"test_snapshot_merge_output_{i}.root"))
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
            for i in range(2):
                p = f"test_snapshot_merge_output_{i}.root"
                if os.path.exists(p):
                    os.remove(p)

    def test_snapshot_merge_and_return_filenames(self):
        """Test Snapshot with merge=True and return_filenames=True."""
        treename = "entries"
        filenames = ["1cluster_20entries.root"] * 2
        backend = DistRDataFrameInvariants.TestBackend()
        headnode = HeadNode.get_headnode(backend, 2, treename, filenames)
        rdf = DataFrame.RDataFrame(headnode)
        output_file = "test_snapshot_merge_return_output.root"
        try:
            files = rdf.Snapshot(treename, output_file, merge=True, return_filenames=True)
            self.assertEqual(files, [output_file])
            self.assertTrue(os.path.exists(output_file))
            for i in range(2):
                self.assertFalse(os.path.exists(f"test_snapshot_merge_return_output_{i}.root"))
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
            for i in range(2):
                p = f"test_snapshot_merge_return_output_{i}.root"
                if os.path.exists(p):
                    os.remove(p)

