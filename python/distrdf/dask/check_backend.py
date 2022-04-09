import os
import pytest

import DistRDF
from DistRDF.Backends import Dask
from DistRDF.Backends.Dask import Backend

import ROOT


class TestDaskBackendInit:
    """
    Tests to ensure that the instance variables of `DaskBackend` class have the
    correct attributes set.
    """

    def test_create_default_backend(self, connection):
        """
        Check that a Dask client object is created with default options for
        the current system.
        """
        backend = Backend.DaskBackend(daskclient=connection)

        assert backend.client is connection

    def test_optimize_npartitions(self, connection):
        """
        Check that `DaskBackend.optimize_npartitions` returns the number of cores
        available to the Dask scheduler.
        """
        backend = Backend.DaskBackend(daskclient=connection)

        assert backend.optimize_npartitions() == 2


class TestInitialization:
    """Check initialization method in the Dask backend"""

    def test_initialization(self, connection):
        """
        Check that the user initialization method is assigned to the current
        backend.
        """
        def returnNumber(n):
            return n

        DistRDF.initialize(returnNumber, 123)

        # Dummy df just to retrieve the initialization function
        df = Dask.RDataFrame(10, daskclient=connection)
        f = df._headnode.backend.initialization

        assert f() == 123

    def test_initialization_method(self, connection):
        """
        Check `DistRDF.initialize` with Dask backend. Defines an integer value
        to the ROOT interpreter. Check that this value is available in the
        worker processes.
        """
        def init(value):
            import ROOT
            cpp_code = f"int userValue = {value};"
            ROOT.gInterpreter.ProcessLine(cpp_code)

        DistRDF.initialize(init, 123)
        # Dask backend has a limited list of supported methods, so we use
        # Histo1D which is a supported action.
        # The code below creates an RDataFrame instance with one single entry
        # and defines a column 'u' whose value is taken from the variable
        # 'userValue'.
        # This variable is only declared inside the ROOT interpreter, however
        # the value of the variable is passed by the user from the python side.
        # If the init function defined by the user is properly propagated to the
        # Dask backend, each workers will run the init function as a first step
        # and hence the variable 'userValue' will be defined at runtime.
        # As a result the define operation should read the variable 'userValue'
        # and assign it to the entries of the column 'u' (only one entry).
        # Finally, Histo1D returns a histogram filled with one value. The mean
        # of this single value has to be the value itself, independently of
        # the number of spawned workers.
        df = Dask.RDataFrame(1, daskclient=connection).Define("u", "userValue").Histo1D(("name", "title", 1, 100, 130), "u")
        h = df.GetValue()
        assert h.GetMean() == 123


class TestEmptyTreeError:
    """
    Tests with emtpy trees.
    """

    def test_histo_from_empty_root_file(self, connection):
        """
        Check that when performing operations with the distributed backend on
        an RDataFrame without entries, DistRDF raises an error.
        """

        # Create an RDataFrame from a file with an empty tree
        rdf = Dask.RDataFrame("NOMINAL", "../emptytree.root", daskclient=connection)
        histo = rdf.Histo1D(("empty", "empty", 10, 0, 10), "mybranch")

        # Get entries in the histogram, raises error
        with pytest.raises(RuntimeError):
            histo.GetEntries()

    def test_count_with_some_empty_trees(self, connection):
        """
        A dataset might contain empty trees. These should be skipped and thus
        not contribute to how many entries are processed in the distributed
        execution.
        """

        opts = ROOT.RDF.RSnapshotOptions()
        opts.fAutoFlush = 10
        df = ROOT.RDataFrame(100).Define("x", "1")
        treenames = [f"tree_{i}" for i in range(3)]
        filenames = [f"distrdf_roottest_dask_check_backend_file_{i}.root" for i in range(3)]
        for treename, filename in zip(treenames, filenames):
            df.Snapshot(treename, filename, ["x"], opts)

        empty_treename = "NOMINAL"
        empty_filename = "../emptytree.root"

        # Create the final dataset with some empty trees
        final_treenames = []
        final_filenames = []
        for treename, filename in zip(treenames, filenames):
            final_treenames.append(empty_treename)
            final_treenames.append(treename)
            final_filenames.append(empty_filename)
            final_filenames.append(filename)

        chain = ROOT.TChain()
        for treename, filename in zip(final_treenames, final_filenames):
            chain.Add(filename + "?#" + treename)

        # 3 files are non-empty, we should always count 300 entries.
        for n in range(1, 6):
            rdf = Dask.RDataFrame(chain, daskclient=connection, npartitions=n)
            assert rdf.Count().GetValue() == 300

        for filename in filenames:
            os.remove(filename)


class TestWithRepeatedTree:
    """
    Tests that the correct number of entries is computed even when the same tree
    is used multiple times.
    """

    def test_count_with_same_tree_repeated(self, connection):
        """
        Count entries of a dataset with three times the same tree.
        """
        df = ROOT.RDataFrame(100).Define("x", "1")
        treename = "tree"
        filename = "distrdf_roottest_dask_check_backend_same_tree.root"
        filenames = [filename] * 3
        df.Snapshot(treename, filename, ["x"])

        rdf = Dask.RDataFrame(treename, filenames, daskclient=connection)
        assert rdf.Count().GetValue() == 300

        os.remove(filename)


class TestChangeAttribute:
    """Tests that check correct changes in the class attributes"""

    def test_user_supplied_npartitions_have_precedence(self, connection):
        """
        The class Client object is connected to a LocalCluster with 2 processes.
        The `DaskBackend.optimize_npartitions` method would thus return 2.
        Check that if the user specifies a number of partitions, that is not
        overwritten by the backend.
        """
        df = Dask.RDataFrame(100, daskclient=connection, npartitions=4)

        # The number of partitions was supplied by the user.
        assert df._headnode.npartitions == 4


if __name__ == "__main__":
    pytest.main(args=[__file__])
