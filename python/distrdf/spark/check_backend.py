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
        df = Spark.RDataFrame(1, sparkcontext=connection).Define("u", "userValue").Histo1D("u")
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
