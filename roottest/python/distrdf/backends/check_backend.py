import platform

import DistRDF
import pytest
import ROOT


class TestBackendInit:
    """
    Tests to ensure that the instance variables of `DaskBackend` class have the
    correct attributes set.
    """

    def test_create_default_backend(self, payload):
        """
        Check that a Dask client object is created with default options for
        the current system.
        """
        connection, backend = payload
        if backend == "dask":
            from DistRDF.Backends.Dask import Backend
            backend = Backend.DaskBackend(daskclient=connection)
            assert backend.client is connection, f"{connection=}"
        elif backend == "spark":
            from DistRDF.Backends.Spark import Backend
            import pyspark
            backend = Backend.SparkBackend()
            assert isinstance(backend.sc, pyspark.SparkContext)

    def test_set_spark_context_with_conf(self, payload):
        """
        Check that a `SparkContext` object is correctly created for a given
        `SparkConf` object in the config dictionary.
        """
        connection, backend = payload
        if backend == "spark":
            from DistRDF.Backends.Spark import Backend
            import pyspark
            backend = Backend.SparkBackend(sparkcontext=connection)

            assert isinstance(backend.sc, pyspark.SparkContext)
            appname = backend.sc.getConf().get("spark.app.name")
            assert appname == "roottest-distrdf-spark"

    def test_optimize_npartitions(self, payload):
        """
        Check that `DaskBackend.optimize_npartitions` returns the number of cores
        available to the Dask scheduler.
        """
        connection, backend = payload
        if backend == "dask":
            from DistRDF.Backends.Dask import Backend
            backend = Backend.DaskBackend(daskclient=connection)
            assert backend.optimize_npartitions() == 2
        elif backend == "spark":
            from DistRDF.Backends.Spark import Backend
            backend = Backend.SparkBackend(sparkcontext=connection)
            assert backend.optimize_npartitions() == 2


class TestInitialization:
    """Check initialization method in the Dask backend"""

    def test_initialization(self, payload):
        """
        Check that the user initialization method is assigned to the current
        backend.
        """
        connection, _ = payload

        def returnNumber(n):
            return n

        DistRDF.initialize(returnNumber, 123)

        df = ROOT.RDataFrame(10, executor=connection)

        # Dummy df just to retrieve the initialization function
        f = df._headnode.backend.initialization

        assert f() == 123

    def test_initialization_method(self, payload):
        """
        Check `DistRDF.initialize` with Dask backend. Defines an integer value
        to the ROOT interpreter. Check that this value is available in the
        worker processes.
        """
        connection, _ = payload

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
        df = ROOT.RDataFrame(1, executor=connection)

        df = df.Define("u", "userValue").Histo1D(
            ("name", "title", 1, 100, 130), "u")
        
        h = df.GetValue()
        assert h.GetMean() == 123


class TestEmptyTreeError:
    """
    Tests with emtpy trees.
    """

    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_histo_from_empty_root_file(self, payload, datasource):
        """
        Check that when performing operations with the distributed backend on
        an RDataFrame without entries, DistRDF raises an error.
        """

        connection, _ = payload
        datasetname = "empty"
        filename = f"../data/{datasource}/empty.root"
        # Create an RDataFrame from a file with an empty tree
        rdf = ROOT.RDataFrame(datasetname, filename, executor=connection)
        histo = rdf.Histo1D(("empty", "empty", 10, 0, 10), "mybranch")

        # Get entries in the histogram, raises error
        with pytest.raises(RuntimeError):
            histo.GetEntries()

    def test_count_with_some_empty_trees(self, payload):
        """
        A dataset might contain empty trees. These should be skipped and thus
        not contribute to how many entries are processed in the distributed
        execution.
        """

        connection, _ = payload
        treenames = [f"tree_{i}" for i in range(3)]
        filenames = [
            f"../data/ttree/distrdf_roottest_check_backend_{i}.root" for i in range(3)]

        empty_treename = "empty"
        empty_filename = "../data/ttree/empty.root"

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
            rdf = ROOT.RDataFrame(chain, executor=connection, npartitions=n)
            assert rdf.Count().GetValue() == 300


class TestWithRepeatedTree:
    """
    Tests that the correct number of entries is computed even when the same tree
    is used multiple times.
    """

    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_count_with_same_tree_repeated(self, payload, datasource):
        """
        Count entries of a dataset with three times the same tree.
        """
        connection, _ = payload
        datasetname = "tree_0"
        filename = f"../data/{datasource}/distrdf_roottest_check_backend_0.root"
        filenames = [filename] * 3

        rdf = ROOT.RDataFrame(datasetname, filenames, executor=connection)

        assert rdf.Count().GetValue() == 300


class TestChangeAttribute:
    """Tests that check correct changes in the class attributes"""

    def test_user_supplied_npartitions_have_precedence(self, payload):
        """
        The class Client object is connected to a LocalCluster with 2 processes.
        The `DaskBackend.optimize_npartitions` method would thus return 2.
        Check that if the user specifies a number of partitions, that is not
        overwritten by the backend.
        """
        connection, _ = payload
        df = ROOT.RDataFrame(100, executor=connection, npartitions=4)

        # The number of partitions was supplied by the user.
        assert df._headnode.npartitions == 4


class TestPropagateExceptions:
    """Tests that the C++ exceptions are properly propagated."""

    @pytest.mark.skipif(platform.system() == "Darwin" and platform.machine() == "arm64",
                        reason="cannot catch exceptions on macOS arm64")
    def test_runtime_error_is_propagated(self, payload):
        """The test creates a TGraph with mixed scalar and vector columns."""
        connection, backend = payload

        df = ROOT.RDataFrame(100, executor=connection)

        if backend == "spark":
            # PySpark raises a custom exception from the RuntimeError raised by
            # DistRDF. Need to make this distinction so that the pytest.raises
            # call does not get confused by the extra level of indirection
            from py4j import protocol
            raised_exc = protocol.Py4JJavaError
        else:
            raised_exc = RuntimeError # Dask always raises a Python RuntimeError

        df = df.Define("x", "1").Define("y", "ROOT::RVecF{1., 2., 3.}")
        g = df.Graph("x", "y")
        cpp_error_what = ("runtime_error: Graph was applied to a mix of scalar "
                          "values and collections. This is not supported.")
        with pytest.raises(raised_exc, match=cpp_error_what):
            g.GetValue()


if __name__ == "__main__":
    pytest.main(args=[__file__])
