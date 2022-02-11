import os
import sys
import unittest
import warnings

import pyspark

import DistRDF
from DistRDF.Backends import Spark
from DistRDF.Backends.Spark import Backend

import ROOT


class SparkBackendInitTest(unittest.TestCase):
    """
    Tests to ensure that the instance variables of `SparkBackend` class have the
    correct attributes set.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Synchronize PYSPARK_PYTHON variable to the current Python executable.
          Needed to avoid mismatch between python versions on driver and on the
          fake executor on the same machine.
        - Ignore `ResourceWarning: unclosed socket` warning triggered by Spark.
          this is ignored by default in any application, but Python's unittest
          library overrides the default warning filters thus exposing this
          warning
        """
        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

    def tearDown(self):
        """Stop any created SparkContext"""
        pyspark.SparkContext.getOrCreate().stop()

    def test_set_spark_context_default(self):
        """
        Check that a `SparkContext` object is created with default options for
        the current system.
        """
        backend = Backend.SparkBackend()

        self.assertIsInstance(backend.sc, pyspark.SparkContext)

    def test_set_spark_context_with_conf(self):
        """
        Check that a `SparkContext` object is correctly created for a given
        `SparkConf` object in the config dictionary.
        """
        conf = {"spark.app.name": "my-pyspark-app1"}
        sconf = pyspark.SparkConf().setAll(conf.items())
        sc = pyspark.SparkContext(conf=sconf)

        backend = Backend.SparkBackend(sparkcontext=sc)

        self.assertIsInstance(backend.sc, pyspark.SparkContext)
        appname = backend.sc.getConf().get("spark.app.name")
        self.assertEqual(appname, "my-pyspark-app1")

    def test_optimize_npartitions(self):
        """
        The optimize_npartitions function returns the value of the
        `defaultParallelism` attribute of the `SparkContext`. This should be
        equal to the number of available cores in case of a context created on
        a single machine.
        """
        ncores = 4
        sconf = pyspark.SparkConf().setMaster(f"local[{ncores}]")
        sc = pyspark.SparkContext(conf=sconf)
        backend = Backend.SparkBackend(sparkcontext=sc)

        self.assertEqual(backend.optimize_npartitions(), ncores)


class InitializationTest(unittest.TestCase):
    """Check initialization method in the Spark backend"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Synchronize PYSPARK_PYTHON variable to the current Python executable.
          Needed to avoid mismatch between python versions on driver and on the
          fake executor on the same machine.
        - Ignore `ResourceWarning: unclosed socket` warning triggered by Spark.
          this is ignored by default in any application, but Python's unittest
          library overrides the default warning filters thus exposing this
          warning
        - Initialize a SparkContext for the tests in this class
        """
        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

        sparkconf = pyspark.SparkConf().setMaster("local[2]")
        cls.sc = pyspark.SparkContext(conf=sparkconf)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

        cls.sc.stop()

    def test_initialization(self):
        """
        Check that the user initialization method is assigned to the current
        backend.
        """
        def returnNumber(n):
            return n

        DistRDF.initialize(returnNumber, 123)

        # Dummy df just to retrieve the initialization function
        df = Spark.RDataFrame(10, sparkcontext=self.sc)
        f = df._headnode.backend.initialization

        self.assertEqual(f(), 123)

    def test_initialization_method(self):
        """
        Check `DistRDF.initialize` with Spark backend. Defines an integer value
        to the ROOT interpreter. Check that this value is available in the
        worker processes.
        """
        def init(value):
            import ROOT
            cpp_code = f"int userValue = {value};"
            ROOT.gInterpreter.ProcessLine(cpp_code)

        DistRDF.initialize(init, 123)
        # Spark backend has a limited list of supported methods, so we use
        # Histo1D which is a supported action.
        # The code below creates an RDataFrame instance with one single entry
        # and defines a column 'u' whose value is taken from the variable
        # 'userValue'.
        # This variable is only declared inside the ROOT interpreter, however
        # the value of the variable is passed by the user from the python side.
        # If the init function defined by the user is properly propagated to the
        # Spark backend, each workers will run the init function as a first step
        # and hence the variable 'userValue' will be defined at runtime.
        # As a result the define operation should read the variable 'userValue'
        # and assign it to the entries of the column 'u' (only one entry).
        # Finally, Histo1D returns a histogram filled with one value. The mean
        # of this single value has to be the value itself, independently of
        # the number of spawned workers.
        df = Spark.RDataFrame(1, sparkcontext=self.sc).Define("u", "userValue").Histo1D("u")
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

        - Synchronize PYSPARK_PYTHON variable to the current Python executable.
          Needed to avoid mismatch between python versions on driver and on the
          fake executor on the same machine.
        - Ignore `ResourceWarning: unclosed socket` warning triggered by Spark.
          this is ignored by default in any application, but Python's unittest
          library overrides the default warning filters thus exposing this
          warning
        - Initialize a SparkContext for the tests in this class
        """
        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

        sparkconf = pyspark.SparkConf().setMaster("local[2]")
        cls.sc = pyspark.SparkContext(conf=sparkconf)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

        cls.sc.stop()

    def test_histo_from_empty_root_file(self):
        """
        Check that when performing operations with the distributed backend on
        an RDataFrame without entries, DistRDF raises an error.
        """

        # Create an RDataFrame from a file with an empty tree
        rdf = Spark.RDataFrame("NOMINAL", "../emptytree.root", sparkcontext=self.sc)
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

        - Synchronize PYSPARK_PYTHON variable to the current Python executable.
          Needed to avoid mismatch between python versions on driver and on the
          fake executor on the same machine.
        - Ignore `ResourceWarning: unclosed socket` warning triggered by Spark.
          this is ignored by default in any application, but Python's unittest
          library overrides the default warning filters thus exposing this
          warning
        - Initialize a SparkContext for the tests in this class
        """
        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

        sparkconf = pyspark.SparkConf().setMaster("local[2]")
        cls.sc = pyspark.SparkContext(conf=sparkconf)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

        cls.sc.stop()

    def test_change_attribute_when_npartitions_greater_than_clusters(self):
        """
        Check that the `npartitions` class attribute is changed when it is
        greater than the number of clusters in the ROOT file.
        """

        treename = "mytree"
        filename = "myfile.root"
        ROOT.RDataFrame(100).Define("x", "rdfentry_").Snapshot(treename, filename)

        df = Spark.RDataFrame(treename, filename, npartitions=10, sparkcontext=self.sc)

        self.assertEqual(df._headnode.npartitions, 10)
        histo = df.Histo1D("x")
        nentries = histo.GetEntries()

        self.assertEqual(nentries, 100)
        self.assertEqual(df._headnode.npartitions, 1)

        os.remove(filename)

    def test_user_supplied_npartitions_have_precedence(self):
        """
        The SparkContext of this class has 2 cores available. The
        `SparkBackend.optimize_npartitions` method would return 2.
        Check that if the user specifies a number of partitions, this
        is not overwritten by the backend.
        """

        df = Spark.RDataFrame(100, sparkcontext=self.sc, npartitions=4)

        # The number of partitions was supplied by the user.
        self.assertEqual(df._headnode.npartitions, 4)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
