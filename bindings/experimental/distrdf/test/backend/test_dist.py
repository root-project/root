import unittest

from DistRDF import DataFrame
from DistRDF import HeadNode
from DistRDF.Backends import Base


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
