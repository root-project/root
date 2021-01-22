import unittest
import PyRDF
from PyRDF.backend.Spark import Spark
from PyRDF.backend.Local import Local
from pyspark import SparkContext


class SelectionTest(unittest.TestCase):
    """Check the accuracy of 'PyRDF.use' method."""

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the `SparkContext` objects that were created during the tests
        in this class.

        """
        context = SparkContext.getOrCreate()
        context.stop()

    def test_spark_select(self):
        """Check if 'spark' environment gets set correctly."""
        PyRDF.use("spark")
        self.assertIsInstance(PyRDF.current_backend, Spark)


class SparkBackendInitTest(unittest.TestCase):
    """
    Tests to ensure that the instance variables
    of `Spark` class are set according to the
    input `config` dict.

    """

    @classmethod
    def tearDown(cls):
        """Clean up the `SparkContext` objects that were created."""
        context = SparkContext.getOrCreate()
        context.stop()

    def test_set_spark_context_default(self):
        """
        Check that if the config dictionary is empty, a `SparkContext`
        object is still created with default options for the current system.

        """
        backend = Spark()

        self.assertDictEqual(backend.config, {})
        self.assertIsInstance(backend.sparkContext, SparkContext)

    def test_set_spark_context_with_conf(self):
        """
        Check that a `SparkContext` object is correctly created for a given
        `SparkConf` object in the config dictionary.

        """
        backend = Spark({'spark.app.name': 'my-pyspark-app1'})

        self.assertIsInstance(backend.sparkContext, SparkContext)
        appname = backend.sparkContext.getConf().get('spark.app.name')
        self.assertEqual(appname, 'my-pyspark-app1')

    def test_set_npartitions_explicit(self):
        """
        Check that the number of partitions is correctly set for a given input
        value in the config dictionary.

        """
        backend = Spark({"npartitions": 5})
        self.assertEqual(backend.npartitions, 5)

    def test_npartitions_with_num_executors(self):
        """
        Check that the number of partitions is correctly set to number of
        executors when no input value is given in the config dictionary.

        """
        backend = Spark({'spark.executor.instances': 10})
        self.assertEqual(backend.npartitions, 10)

    def test_npartitions_with_already_existing_spark_context(self):
        """
        Check that the number of partitions is correctly set when a Spark
        Context already exists.

        """
        from pyspark import SparkConf
        sparkConf = SparkConf().set('spark.executor.instances', 15)
        SparkContext(conf=sparkConf)
        backend = Spark()
        self.assertEqual(backend.npartitions, 15)

    def test_npartitions_default(self):
        """
        Check that the default number of partitions is correctly set when no
        input value is given in the config dictionary.

        """
        backend = Spark()
        self.assertEqual(backend.npartitions, Spark.MIN_NPARTITIONS)


class OperationSupportTest(unittest.TestCase):
    """
    Ensure that incoming operations are classified accurately in distributed
    environment.

    """

    @classmethod
    def tearDown(cls):
        """Clean up the `SparkContext` objects that were created."""
        context = SparkContext.getOrCreate()
        context.stop()

    def test_action(self):
        """Check that action nodes are classified accurately."""
        backend = Spark()
        backend.check_supported("Histo1D")

    def test_transformation(self):
        """Check that transformation nodes are classified accurately."""
        backend = Spark()
        backend.check_supported("Define")

    def test_unsupported_operations(self):
        """Check that unsupported operations raise an Exception."""
        backend = Spark()
        with self.assertRaises(Exception):
            backend.check_supported("Take")

        with self.assertRaises(Exception):
            backend.check_supported("Foreach")

        with self.assertRaises(Exception):
            backend.check_supported("Range")

    def test_none(self):
        """Check that incorrect operations raise an Exception."""
        backend = Spark()
        with self.assertRaises(Exception):
            backend.check_supported("random")

    def test_range_operation_single_thread(self):
        """
        Check that 'Range' operation works in single-threaded mode and raises an
        Exception in multi-threaded mode.

        """
        backend = Spark()
        with self.assertRaises(Exception):
            backend.check_supported("Range")


class InitializationTest(unittest.TestCase):
    """Check initialization method in the Spark backend"""

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

        PyRDF.initialize(init, 123)
        PyRDF.current_backend = Spark()
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
        df = PyRDF.RDataFrame(1).Define("u", "userValue").Histo1D("u")
        h = df.GetValue()
        self.assertEqual(h.GetMean(), 123)


class FallbackTest(unittest.TestCase):
    """
    Check cases when the distributed backend has to fallback to local execution
    """

    def test_histo_from_empty_root_file(self):
        """
        Check that when performing operations with the distributed backend on
        an RDataFrame without entries, PyRDF falls back to using the local
        backend and outputs the correct (empty) result.
        """
        PyRDF.use("spark")

        # Creates and RDataFrame with 10 integers [0...9]
        rdf = PyRDF.RDataFrame("NOMINAL", "emptytree.root")
        histo = rdf.Histo1D("mybranch")

        # Get entries in the histogram, should be zero
        entries = histo.GetEntries()

        self.assertIsInstance(PyRDF.current_backend, Local)
        self.assertEqual(entries, 0)


class ChangeAttributeTest(unittest.TestCase):
    """Tests that check correct changes in the class attributes"""

    def test_change_attribute_when_npartitions_greater_than_clusters(self):
        """
        Check that the `npartitions class attribute is changed when it is
        greater than the number of clusters in the ROOT file.
        """
        PyRDF.use("spark", {"npartitions": 10})

        from PyRDF import current_backend

        self.assertEqual(current_backend.npartitions, 10)

        treename = "TotemNtuple"
        filelist = ["Slimmed_ntuple.root"]
        df = PyRDF.RDataFrame(treename, filelist)

        histo = df.Histo1D("track_rp_3.x")
        nentries = histo.GetEntries()

        self.assertEqual(nentries, 10)
        self.assertEqual(current_backend.npartitions, 1)


if __name__ == "__main__":
    unittest.main()
