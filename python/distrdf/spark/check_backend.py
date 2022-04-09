import os

import pyspark

import pytest

import DistRDF
from DistRDF.Backends import Spark
from DistRDF.Backends.Spark import Backend

import ROOT


class TestSparkBackendInit:
    """
    Tests to ensure that the instance variables of `SparkBackend` class have the
    correct attributes set.
    """

    def test_set_spark_context_default(self, connection):
        """
        Check that a `SparkContext` object is created with default options for
        the current system.
        """
        backend = Backend.SparkBackend()

        assert isinstance(backend.sc, pyspark.SparkContext)

    def test_set_spark_context_with_conf(self, connection):
        """
        Check that a `SparkContext` object is correctly created for a given
        `SparkConf` object in the config dictionary.
        """
        backend = Backend.SparkBackend(sparkcontext=connection)

        assert isinstance(backend.sc, pyspark.SparkContext)
        appname = backend.sc.getConf().get("spark.app.name")
        assert appname == "roottest-distrdf-spark"

    def test_optimize_npartitions(self, connection):
        """
        The optimize_npartitions function returns the value of the
        `defaultParallelism` attribute of the `SparkContext`. This should be
        equal to the number of available cores in case of a context created on
        a single machine.
        """

        backend = Backend.SparkBackend(sparkcontext=connection)

        assert backend.optimize_npartitions() == 2


class TestInitialization:
    """Check initialization method in the Spark backend"""

    def test_initialization(self, connection):
        """
        Check that the user initialization method is assigned to the current
        backend.
        """
        def returnNumber(n):
            return n

        DistRDF.initialize(returnNumber, 123)

        # Dummy df just to retrieve the initialization function
        df = Spark.RDataFrame(10, sparkcontext=connection)
        f = df._headnode.backend.initialization

        assert f() == 123

    def test_initialization_method(self, connection):
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
        df = Spark.RDataFrame(1, sparkcontext=connection).Define("u", "userValue").Histo1D(("name", "title", 1, 100, 130), "u")
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
        rdf = Spark.RDataFrame("NOMINAL", "../emptytree.root", sparkcontext=connection)
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
        filenames = [f"distrdf_roottest_spark_check_backend_file_{i}.root" for i in range(3)]
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
            rdf = Spark.RDataFrame(chain, sparkcontext=connection, npartitions=n)
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
        filename = "distrdf_roottest_spark_check_backend_same_tree.root"
        filenames = [filename] * 3
        df.Snapshot(treename, filename, ["x"])

        rdf = Spark.RDataFrame(treename, filenames, sparkcontext=connection)
        assert rdf.Count().GetValue() == 300

        os.remove(filename)


class TestChangeAttribute:
    """Tests that check correct changes in the class attributes"""

    def test_user_supplied_npartitions_have_precedence(self, connection):
        """
        The SparkContext of this class has 2 cores available. The
        `SparkBackend.optimize_npartitions` method would return 2.
        Check that if the user specifies a number of partitions, this
        is not overwritten by the backend.
        """

        df = Spark.RDataFrame(100, sparkcontext=connection, npartitions=4)

        # The number of partitions was supplied by the user.
        assert df._headnode.npartitions == 4


if __name__ == "__main__":
    pytest.main(args=[__file__])
