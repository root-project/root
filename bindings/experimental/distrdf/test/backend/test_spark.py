import os
import sys
import unittest
import warnings

import DistRDF
import pyspark
from DistRDF.Backends import Spark
from DistRDF.Backends.Spark import Backend


class SparkBackendInitTest(unittest.TestCase):
    """
    Tests to ensure that the instance variables
    of `Spark` class are set according to the
    input `config` dict.
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
        os.environ["PYSPARK_PYTHON"] = sys.executable

        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        os.environ["PYSPARK_PYTHON"] = ""

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

    def test_optimize_npartitions_with_num_executors(self):
        """
        Check that the number of partitions is correctly set to number of
        executors in the SparkConf dictionary.
        """
        conf = {"spark.executor.instances": 10}
        sconf = pyspark.SparkConf().setAll(conf.items())
        sc = pyspark.SparkContext(conf=sconf)
        backend = Backend.SparkBackend(sparkcontext=sc)

        self.assertEqual(backend.optimize_npartitions(), 10)


class OperationSupportTest(unittest.TestCase):
    """
    Ensure that incoming operations are classified accurately in distributed
    environment.
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
        os.environ["PYSPARK_PYTHON"] = sys.executable

        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        os.environ["PYSPARK_PYTHON"] = ""

        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

    def tearDown(self):
        """Stop any created SparkContext"""
        pyspark.SparkContext.getOrCreate().stop()

    def test_action(self):
        """Check that action nodes are classified accurately."""
        backend = Backend.SparkBackend()
        backend.check_supported("Histo1D")

    def test_transformation(self):
        """Check that transformation nodes are classified accurately."""
        backend = Backend.SparkBackend()
        backend.check_supported("Define")

    def test_unsupported_operations(self):
        """Check that unsupported operations raise an Exception."""
        backend = Backend.SparkBackend()
        with self.assertRaises(Exception):
            backend.check_supported("Take")

        with self.assertRaises(Exception):
            backend.check_supported("Foreach")

        with self.assertRaises(Exception):
            backend.check_supported("Range")

    def test_none(self):
        """Check that incorrect operations raise an Exception."""
        backend = Backend.SparkBackend()
        with self.assertRaises(Exception):
            backend.check_supported("random")

    def test_range_operation_single_thread(self):
        """
        Check that 'Range' operation works in single-threaded mode and raises an
        Exception in multi-threaded mode.
        """
        backend = Backend.SparkBackend()
        with self.assertRaises(Exception):
            backend.check_supported("Range")


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
        """
        os.environ["PYSPARK_PYTHON"] = sys.executable

        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        os.environ["PYSPARK_PYTHON"] = ""

        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

    def tearDown(self):
        """Stop any created SparkContext"""
        pyspark.SparkContext.getOrCreate().stop()

    def test_initialization(self):
        """
        Check that the user initialization method is assigned to the current
        backend.
        """
        def returnNumber(n):
            return n

        DistRDF.initialize(returnNumber, 123)

        # Dummy df just to retrieve the initialization function
        df = Spark.RDataFrame(10)
        f = df._headnode.backend.initialization

        self.assertEqual(f(), 123)

    def test_initialization_method(self):
        """
        Check initialization method in Spark backend.
        Define a method in the ROOT interpreter called getValue which returns
        the value defined by the user on the python side.
        """
        def init(value):
            import ROOT
            cpp_code = '''int userValue = %s ;''' % value
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
        df = Spark.RDataFrame(1).Define(
            "u", "userValue").Histo1D("u")
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
        """
        os.environ["PYSPARK_PYTHON"] = sys.executable

        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        os.environ["PYSPARK_PYTHON"] = ""

        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

    def tearDown(self):
        """Stop any created SparkContext"""
        pyspark.SparkContext.getOrCreate().stop()

    def test_histo_from_empty_root_file(self):
        """
        Check that when performing operations with the distributed backend on
        an RDataFrame without entries, DistRDF raises an error.
        """

        # Create an RDataFrame from a file with an empty tree
        rdf = Spark.RDataFrame(
            "NOMINAL", "emptytree.root")
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
        """
        os.environ["PYSPARK_PYTHON"] = sys.executable

        if sys.version_info.major >= 3:
            warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        os.environ["PYSPARK_PYTHON"] = ""

        if sys.version_info.major >= 3:
            warnings.simplefilter("default", ResourceWarning)

    def tearDown(self):
        """Stop any created SparkContext"""
        pyspark.SparkContext.getOrCreate().stop()

    def test_change_attribute_when_npartitions_greater_than_clusters(self):
        """
        Check that the `npartitions class attribute is changed when it is
        greater than the number of clusters in the ROOT file.
        """

        treename = "TotemNtuple"
        filelist = ["Slimmed_ntuple.root"]
        df = Spark.RDataFrame(treename, filelist, npartitions=10)

        self.assertEqual(df._headnode.npartitions, 10)
        histo = df.Histo1D("track_rp_3.x")
        nentries = histo.GetEntries()

        self.assertEqual(nentries, 10)
        self.assertEqual(df._headnode.npartitions, 1)

    def test_optimize_npartitions_with_spark_config_options(self):
        """
        Check that relevant spark configuration options optimize the number of
        partitions.
        """

        conf = {"spark.executor.cores": 4, "spark.executor.instances": 4}
        sconf = pyspark.SparkConf().setAll(conf.items())
        scontext = pyspark.SparkContext(conf=sconf)

        df = Spark.RDataFrame(100, sparkcontext=scontext)

        # The number of partitions was optimized to be equal to
        # spark.executor.cores * spark.executor.instances
        self.assertEqual(df._headnode.npartitions, 16)

    def test_user_supplied_npartitions_have_precedence(self):
        """
        Check that even if spark configuration options could optimize the number
        of partitions, a user supplied value for npartitions takes precedence.
        """

        conf = {"spark.executor.cores": 4, "spark.executor.instances": 4}
        sconf = pyspark.SparkConf().setAll(conf.items())
        scontext = pyspark.SparkContext(conf=sconf)

        df = Spark.RDataFrame(100, sparkcontext=scontext, npartitions=4)

        # The number of partitions was supplied by the user.
        self.assertEqual(df._headnode.npartitions, 4)


if __name__ == "__main__":
    unittest.main()
