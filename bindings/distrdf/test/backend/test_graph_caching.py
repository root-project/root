import os
import unittest
from array import array

import ROOT
from DistRDF._graph_cache import _ACTIONS_REGISTER, _RDF_REGISTER
from DistRDF.Backends import Base
from DistRDF.DataFrame import RDataFrame
from DistRDF.HeadNode import get_headnode


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

    def _test_snapshot_impl(self, backend, treename, filenames, nentries):
        for npartitions in [1, 2, 4, 8, 16]:
            # Start from a fresh cache at each subtest iteration
            clear_caches()

            output_basename = "test_graph_caching_test_snapshot"
            output_filenames = [f"{output_basename}_{i}.root" for i in range(npartitions)]
            try:
                with self.subTest(npartitions=npartitions):
                    headnode = get_headnode(backend, npartitions, treename, filenames)
                    distrdf = RDataFrame(headnode)

                    output_branch = "b1"
                    snapdf = distrdf.Snapshot(treename, f"{output_basename}.root", (output_branch,))

                    # There should be exactly one cached RDF and set of actions
                    self.assertEqual(len(_RDF_REGISTER), 1)
                    self.assertEqual(len(_RDF_REGISTER), len(_ACTIONS_REGISTER))
                    cached_rdf = tuple(_RDF_REGISTER.values())[0]
                    # The RDataFrame should have run as many times as partitions
                    self.assertEqual(cached_rdf.GetNRuns(), npartitions)
                    # All the correct output files should be present
                    for output_filename in output_filenames:
                        self.assertTrue(os.path.exists(output_filename))
                    # Make sure we have the correct data in the snapshot output
                    self.assertListEqual(list(snapdf.AsNumpy()[output_branch]), list(range(500)))
            finally:
                # Remove output files at each iteration
                for output_filename in output_filenames:
                    try:
                        os.remove(output_filename)
                    except OSError:
                        pass

    def test_snapshot(self):
        """The cache is used to Snapshot data."""

        def write_data(treenames, filenames):
            # Create a dataset with 100 entries per file with sequential values
            b1 = array("i", [0])
            for i in range(1, len(filenames) + 1):
                with ROOT.TFile.Open(filenames[i - 1], "recreate") as f:
                    t = ROOT.TTree(treenames[i - 1], treenames[i - 1])
                    t.Branch("b1", b1, "b1/I")
                    nentries = 0
                    for val in range((i - 1) * 100, i * 100):
                        b1[0] = val
                        t.Fill()
                        nentries += 1
                        # 5 clusters per file
                        if nentries % 20 == 0:
                            t.FlushBaskets()
                    f.Write()

        treename = "Events"
        filenames = [f"test_graph_caching_test_snapshot_input_{i}.root" for i in range(1, 6)]
        nentries = 100 * len(filenames)
        backend = GraphCaching.TestBackend()

        try:
            write_data([treename] * len(filenames), filenames)
            self._test_snapshot_impl(backend, treename, filenames, nentries)
        finally:
            for fn in filenames:
                try:
                    os.remove(fn)
                except OSError:
                    pass

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
