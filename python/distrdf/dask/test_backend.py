import os
import unittest

from dask.distributed import Client, LocalCluster

import DistRDF
from DistRDF.Backends import Dask
from DistRDF.Backends.Dask import Backend

import ROOT


class DaskBackendInitTest(unittest.TestCase):
    """
    Tests to ensure that the instance variables of `DaskBackend` class have the
    correct attributes set.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

    def test_create_default_backend(self):
        """
        Check that a Dask client object is created with default options for
        the current system.
        """
        backend = Backend.DaskBackend(daskclient=self.client)

        self.assertTrue(backend.client is self.client)

    def test_optimize_npartitions(self):
        """
        Check that `DaskBackend.optimize_npartitions` returns the number of cores
        available to the Dask scheduler.
        """
        backend = Backend.DaskBackend(daskclient=self.client)

        self.assertEqual(backend.optimize_npartitions(), 2)


class OperationSupportTest(unittest.TestCase):
    """
    Ensure that incoming operations are classified accurately in distributed
    environment.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

    def test_action(self):
        """Check that action nodes are classified accurately."""
        backend = Backend.DaskBackend(daskclient=self.client)
        backend.check_supported("Histo1D")

    def test_transformation(self):
        """Check that transformation nodes are classified accurately."""
        backend = Backend.DaskBackend(daskclient=self.client)
        backend.check_supported("Define")

    def test_unsupported_operations(self):
        """Check that unsupported operations raise an Exception."""
        backend = Backend.DaskBackend(daskclient=self.client)
        with self.assertRaises(Exception):
            backend.check_supported("Take")

        with self.assertRaises(Exception):
            backend.check_supported("Foreach")

        with self.assertRaises(Exception):
            backend.check_supported("Range")

    def test_none(self):
        """Check that incorrect operations raise an Exception."""
        backend = Backend.DaskBackend(daskclient=self.client)
        with self.assertRaises(Exception):
            backend.check_supported("random")


class InitializationTest(unittest.TestCase):
    """Check initialization method in the Dask backend"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

    def test_initialization(self):
        """
        Check that the user initialization method is assigned to the current
        backend.
        """
        def returnNumber(n):
            return n

        DistRDF.initialize(returnNumber, 123)

        # Dummy df just to retrieve the initialization function
        df = Dask.RDataFrame(10, daskclient=self.client)
        f = df._headnode.backend.initialization

        self.assertEqual(f(), 123)

    def test_initialization_method(self):
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
        df = Dask.RDataFrame(1, daskclient=self.client).Define("u", "userValue").Histo1D("u")
        h = df.GetValue()
        self.assertEqual(h.GetMean(), 123)


class EmptyTreeErrorTest(unittest.TestCase):
    """
    Distributed execution fails when the tree has no entries.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

    def test_histo_from_empty_root_file(self):
        """
        Check that when performing operations with the distributed backend on
        an RDataFrame without entries, DistRDF raises an error.
        """

        # Create an RDataFrame from a file with an empty tree
        rdf = Dask.RDataFrame("NOMINAL", "../emptytree.root", daskclient=self.client)
        histo = rdf.Histo1D(("empty", "empty", 10, 0, 10), "mybranch")

        # Get entries in the histogram, raises error
        with self.assertRaises(RuntimeError):
            histo.GetEntries()


class ChangeAttributeTest(unittest.TestCase):
    """Tests that check correct changes in the class attributes"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

    def test_change_attribute_when_npartitions_greater_than_clusters(self):
        """
        Check that the `npartitions` class attribute is changed when it is
        greater than the number of clusters in the ROOT file.
        """

        treename = "mytree"
        filename = "myfile.root"
        ROOT.RDataFrame(100).Define("x", "rdfentry_").Snapshot(treename, filename)

        df = Dask.RDataFrame(treename, filename, npartitions=10, daskclient=self.client)

        self.assertEqual(df._headnode.npartitions, 10)
        histo = df.Histo1D("x")
        nentries = histo.GetEntries()

        self.assertEqual(nentries, 100)
        self.assertEqual(df._headnode.npartitions, 1)

        os.remove(filename)

    def test_user_supplied_npartitions_have_precedence(self):
        """
        The class Client object is connected to a LocalCluster with 2 processes.
        The `DaskBackend.optimize_npartitions` method would thus return 2.
        Check that if the user specifies a number of partitions, that is not
        overwritten by the backend.
        """

        df = Dask.RDataFrame(100, daskclient=self.client, npartitions=4)

        # The number of partitions was supplied by the user.
        self.assertEqual(df._headnode.npartitions, 4)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
