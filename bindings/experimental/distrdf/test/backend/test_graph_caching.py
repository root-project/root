
import os
import unittest

from DistRDF.Backends import Base
from DistRDF.DataFrame import RDataFrame
from DistRDF.HeadNode import get_headnode
from DistRDF._graph_cache import _ACTIONS_REGISTER, _RDF_REGISTER


def clear_caches():
    _RDF_REGISTER.clear()
    _ACTIONS_REGISTER.clear()


class GraphCaching(unittest.TestCase):
    """
    Different tasks run in the same process should not need to re-JIT the
    computation graph.
    """

    class TestBackend(Base.BaseBackend):

        def ProcessAndMerge(self, ranges, mapper, reducer):
            mergeables_lists = [mapper(range) for range in ranges]

            while len(mergeables_lists) > 1:
                mergeables_lists.append(
                    reducer(mergeables_lists.pop(0), mergeables_lists.pop(0)))

            return mergeables_lists.pop()

        def distribute_unique_paths(self, includes_list): ...

        def make_dataframe(self, *args, **kwargs):
            """This implementation is needed for the Snapshot tests."""
            npartitions = 1
            headnode = get_headnode(self, npartitions, *args)
            return RDataFrame(headnode)

        def optimize_npartitions(self): ...

    def tearDown(self) -> None:
        """Clear the caches in between tests."""
        clear_caches()

    def test_count_emptysource(self):
        """The cache is used to count entries with an empty source."""
        nentries = 100
        backend = GraphCaching.TestBackend()

        for npartitions in range(1, 11):
            # Start from a fresh cache at each subtest iteration
            clear_caches()
            with self.subTest(npartitions=npartitions):
                headnode = get_headnode(backend, npartitions, nentries)
                distrdf = RDataFrame(headnode)
                # The count operation should always return the correct value
                self.assertEqual(distrdf.Count().GetValue(), 100)
                # There should be exactly one cached RDF and set of actions
                self.assertEqual(len(_RDF_REGISTER), 1)
                self.assertEqual(len(_RDF_REGISTER), len(_ACTIONS_REGISTER))
                cached_rdf = tuple(_RDF_REGISTER.values())[0]
                # The RDataFrame should have run as many times as partitions
                self.assertEqual(cached_rdf.GetNRuns(), npartitions)

    def test_count_ttree(self):
        """The cache is used to count entries of a TTree."""
        treename = "myTree"
        filename = "4clusters.root"
        nentries = 1000
        backend = GraphCaching.TestBackend()

        # The maximum number of partitions is the number of clusters
        for npartitions in range(1, 4):
            # Start from a fresh cache at each subtest iteration
            clear_caches()
            with self.subTest(npartitions=npartitions):
                headnode = get_headnode(backend, npartitions, treename, filename)
                distrdf = RDataFrame(headnode)
                # The count operation should always return the correct value
                self.assertEqual(distrdf.Count().GetValue(), nentries)
                # There should be exactly one cached RDF and set of actions
                self.assertEqual(len(_RDF_REGISTER), 1)
                self.assertEqual(len(_RDF_REGISTER), len(_ACTIONS_REGISTER))
                cached_rdf = tuple(_RDF_REGISTER.values())[0]
                # The RDataFrame should have run as many times as partitions
                self.assertEqual(cached_rdf.GetNRuns(), npartitions)

    def test_count_tchain(self):
        """The cache is used to count entries of a TChain."""
        treename = "myTree"
        filenames = ["4clusters.root"] * 5
        nentries = 5000
        backend = GraphCaching.TestBackend()

        # The maximum number of partitions is the number of clusters
        for npartitions in range(1, 20):
            # Start from a fresh cache at each subtest iteration
            clear_caches()
            with self.subTest(npartitions=npartitions):
                headnode = get_headnode(backend, npartitions, treename, filenames)
                distrdf = RDataFrame(headnode)
                # The count operation should always return the correct value
                self.assertEqual(distrdf.Count().GetValue(), nentries)
                # There should be exactly one cached RDF and set of actions
                self.assertEqual(len(_RDF_REGISTER), 1)
                self.assertEqual(len(_RDF_REGISTER), len(_ACTIONS_REGISTER))
                cached_rdf = tuple(_RDF_REGISTER.values())[0]
                # The RDataFrame should have run as many times as partitions
                self.assertEqual(cached_rdf.GetNRuns(), npartitions)

    def test_snapshot(self):
        """The cache is used to Snapshot data."""
        treename = "myTree"
        filenames = ["4clusters.root"] * 5
        nentries = 5000
        backend = GraphCaching.TestBackend()

        for npartitions in [1, 2, 4, 8, 16]:
            # Start from a fresh cache at each subtest iteration
            clear_caches()

            output_basename = "test_graph_caching_test_snapshot"
            output_filenames = [f"{output_basename}_{i}.root" for i in range(npartitions)]

            with self.subTest(npartitions=npartitions):
                headnode = get_headnode(backend, npartitions, treename, filenames)
                distrdf = RDataFrame(headnode)

                output_branches = ("b1", )
                snapdf = distrdf.Snapshot(treename, f"{output_basename}.root", ["b1", ])

                # There should be exactly one cached RDF and set of actions
                self.assertEqual(len(_RDF_REGISTER), 1)
                self.assertEqual(len(_RDF_REGISTER), len(_ACTIONS_REGISTER))
                cached_rdf = tuple(_RDF_REGISTER.values())[0]
                # The RDataFrame should have run as many times as partitions
                self.assertEqual(cached_rdf.GetNRuns(), npartitions)
                # All the correct output files should be present
                for output_filename in output_filenames:
                    self.assertTrue(os.path.exists(output_filename))
                # The snapshotted dataframe should be usable
                self.assertEqual(snapdf.Count().GetValue(), nentries)

            # Remove output files at each iteration
            for output_filename in output_filenames:
                os.remove(output_filename)

    def test_multiple_graphs(self):
        """The caches are used with multiple executions."""
        treename = "myTree"
        filenames = ["4clusters.root"] * 5
        nentries = 5000
        backend = GraphCaching.TestBackend()

        for npartitions in [1, 2, 4, 8, 16]:
            # Start from a fresh cache at each subtest iteration
            clear_caches()
            with self.subTest(npartitions=npartitions):
                headnode = get_headnode(backend, npartitions, treename, filenames)
                distrdf = RDataFrame(headnode)

                self.assertEqual(distrdf.Count().GetValue(), nentries)

                distrdf = distrdf.Define("x", "2").Define("y", "3").Define("z", "x*y")

                sumx = distrdf.Sum("x")
                sumy = distrdf.Sum("y")

                self.assertEqual(sumx.GetValue(), 10000)
                self.assertEqual(sumy.GetValue(), 15000)

                self.assertEqual(len(_RDF_REGISTER), 2)
                self.assertEqual(len(_ACTIONS_REGISTER), len(_RDF_REGISTER))

                for cached_rdf in _RDF_REGISTER.values():
                    self.assertEqual(cached_rdf.GetNRuns(), npartitions)

                meanz = distrdf.Mean("z")

                self.assertAlmostEqual(meanz.GetValue(), 6)

                self.assertEqual(len(_RDF_REGISTER), 3)
                self.assertEqual(len(_ACTIONS_REGISTER), len(_RDF_REGISTER))

                for cached_rdf in _RDF_REGISTER.values():
                    self.assertEqual(cached_rdf.GetNRuns(), npartitions)


if __name__ == "__main__":
    unittest.main()
