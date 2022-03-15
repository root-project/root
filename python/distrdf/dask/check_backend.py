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
        df = Dask.RDataFrame(1, daskclient=connection).Define("u", "userValue").Histo1D("u")
        h = df.GetValue()
        assert h.GetMean() == 123


class TestEmptyTreeError:
    """
    Distributed execution fails when the tree has no entries.
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


class TestChangeAttribute:
    """Tests that check correct changes in the class attributes"""

    def test_change_attribute_when_npartitions_greater_than_clusters(self, connection):
        """
        Check that the `npartitions` class attribute is changed when it is
        greater than the number of clusters in the ROOT file.
        """

        treename = "mytree"
        filename = "myfile.root"
        ROOT.RDataFrame(100).Define("x", "rdfentry_").Snapshot(treename, filename)

        df = Dask.RDataFrame(treename, filename, npartitions=10, daskclient=connection)

        assert df._headnode.npartitions == 10
        histo = df.Histo1D("x")
        nentries = histo.GetEntries()

        assert nentries == 100
        assert df._headnode.npartitions == 1

        os.remove(filename)

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
