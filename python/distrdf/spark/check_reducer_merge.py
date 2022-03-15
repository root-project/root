import os

import pytest

import numpy

import ROOT

from DistRDF.Backends import Spark
from DistRDF.Proxy import ActionProxy


class TestReducerMerge:
    """Check the working of merge operations in the reducer function."""

    def assertHistoOrProfile(self, obj_1, obj_2):
        """Asserts equality between two 'ROOT.TH1' or 'ROOT.TH2' objects."""
        # Compare the sizes of equivalent objects
        assert obj_1.GetEntries() == obj_2.GetEntries()

        # Compare the means of equivalent objects
        assert obj_1.GetMean() == obj_2.GetMean()

        # Compare the standard deviations of equivalent objects
        assert obj_1.GetStdDev() == obj_2.GetStdDev()

    def define_two_columns(self, rdf):
        """
        Helper method that Defines and returns two columns with definitions
        "x = rdfentry_" and "y = rdfentry_ * rdfentry_".

        """
        return rdf.Define("x", "rdfentry_").Define("y", "rdfentry_*rdfentry_")

    def define_three_columns(self, rdf):
        """
        Helper method that Defines and returns three columns with definitions
        "x = rdfentry_", "y = rdfentry_ * rdfentry_" and
        "z = rdfentry_ * rdfentry_ * rdfentry_".

        """
        return rdf.Define("x", "rdfentry_")\
                  .Define("y", "rdfentry_*rdfentry_")\
                  .Define("z", "rdfentry_*rdfentry_*rdfentry_")

    def define_four_columns(self, rdf, colnames):
        """Helper method to define four columns."""
        for name in colnames:
            rdf = rdf.Define(name, "rdfentry_")

        return rdf

    def test_histo1d_merge(self, connection):
        """Check the working of Histo1D merge operation in the reducer."""
        # Operations with DistRDF
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        histo_py = rdf_py.Histo1D("rdfentry_")

        # Operations with PyROOT
        rdf_cpp = ROOT.ROOT.RDataFrame(10)
        histo_cpp = rdf_cpp.Histo1D("rdfentry_")

        # Compare the 2 histograms
        self.assertHistoOrProfile(histo_py, histo_cpp)

    def test_histo2d_merge(self, connection):
        """Check the working of Histo2D merge operation in the reducer."""
        modelTH2D = ("", "", 64, -4, 4, 64, -4, 4)

        # Operations with DistRDF
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        columns_py = self.define_two_columns(rdf_py)
        histo_py = columns_py.Histo2D(modelTH2D, "x", "y")

        # Operations with PyROOT
        rdf_cpp = ROOT.ROOT.RDataFrame(10)
        columns_cpp = self.define_two_columns(rdf_cpp)
        histo_cpp = columns_cpp.Histo2D(modelTH2D, "x", "y")

        # Compare the 2 histograms
        self.assertHistoOrProfile(histo_py, histo_cpp)

    def test_histo3d_merge(self, connection):
        """Check the working of Histo3D merge operation in the reducer."""
        modelTH3D = ("", "", 64, -4, 4, 64, -4, 4, 64, -4, 4)
        # Operations with DistRDF
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        columns_py = self.define_three_columns(rdf_py)
        histo_py = columns_py.Histo3D(modelTH3D, "x", "y", "z")

        # Operations with PyROOT
        rdf_cpp = ROOT.ROOT.RDataFrame(10)
        columns_cpp = self.define_three_columns(rdf_cpp)
        histo_cpp = columns_cpp.Histo3D(modelTH3D, "x", "y", "z")

        # Compare the 2 histograms
        self.assertHistoOrProfile(histo_py, histo_cpp)

    def test_histond_merge(self, connection):
        """Check the working of HistoND merge operation in the reducer."""
        nbins = (10, 10, 10, 10)
        xmin = (0., 0., 0., 0.)
        xmax = (100., 100., 100., 100.)
        modelTHND = ("name", "title", 4, nbins, xmin, xmax)
        colnames = ("x0", "x1", "x2", "x3")

        distrdf = Spark.RDataFrame(100, sparkcontext=connection)
        rdf = ROOT.RDataFrame(100)

        distrdf_withcols = self.define_four_columns(distrdf, colnames)
        rdf_withcols = self.define_four_columns(rdf, colnames)

        histond_distrdf = distrdf_withcols.HistoND(modelTHND, colnames)
        histond_rdf = rdf_withcols.HistoND(modelTHND, colnames)

        assert histond_distrdf.GetEntries() == histond_rdf.GetEntries()
        assert histond_distrdf.GetNbins() == histond_rdf.GetNbins()

    def test_profile1d_merge(self, connection):
        """Check the working of Profile1D merge operation in the reducer."""
        # Operations with DistRDF
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        columns_py = self.define_two_columns(rdf_py)
        profile_py = columns_py.Profile1D(("", "", 64, -4, 4), "x", "y")

        # Operations with PyROOT
        rdf_cpp = ROOT.ROOT.RDataFrame(10)
        columns_cpp = self.define_two_columns(rdf_cpp)
        profile_cpp = columns_cpp.Profile1D(("", "", 64, -4, 4), "x", "y")

        # Compare the 2 profiles
        self.assertHistoOrProfile(profile_py, profile_cpp)

    def test_profile2d_merge(self, connection):
        """Check the working of Profile2D merge operation in the reducer."""
        model = ("", "", 64, -4, 4, 64, -4, 4)

        # Operations with DistRDF
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        columns_py = self.define_three_columns(rdf_py)
        profile_py = columns_py.Profile2D(model, "x", "y", "z")

        # Operations with PyROOT
        rdf_cpp = ROOT.ROOT.RDataFrame(10)
        columns_cpp = self.define_three_columns(rdf_cpp)
        profile_cpp = columns_cpp.Profile2D(model, "x", "y", "z")

        # Compare the 2 profiles
        self.assertHistoOrProfile(profile_py, profile_cpp)

    def test_tgraph_merge(self, connection):
        """Check the working of TGraph merge operation in the reducer."""
        # Operations with DistRDF
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        columns_py = self.define_two_columns(rdf_py)
        graph_py = columns_py.Graph("x", "y")

        # Operations with PyROOT
        rdf_cpp = ROOT.ROOT.RDataFrame(10)
        columns_cpp = self.define_two_columns(rdf_cpp)
        graph_cpp = columns_cpp.Graph("x", "y")

        # Sort the graphs to make sure corresponding points are same
        graph_py.Sort()
        graph_cpp.Sort()

        # Compare the X co-ordinates of the graphs
        assert list(graph_py.GetX()) == list(graph_cpp.GetX())

        # Compare the Y co-ordinates of the graphs
        assert list(graph_py.GetY()) == list(graph_cpp.GetY())

    def test_distributed_count(self, connection):
        """Test support for `Count` operation in distributed backend"""
        rdf_py = Spark.RDataFrame(100, sparkcontext=connection)
        count = rdf_py.Count()

        assert count.GetValue() == 100

    def test_distributed_sum(self, connection):
        """Test support for `Sum` operation in distributed backend"""
        rdf_py = Spark.RDataFrame(10, sparkcontext=connection)
        rdf_def = rdf_py.Define("x", "rdfentry_")
        rdf_sum = rdf_def.Sum("x")

        assert rdf_sum.GetValue() == 45.0

    def check_npy_dict(self, npy_dict):
        """Checks on correctness of numpy array dictionary returned by 'Asnumpy'"""
        assert isinstance(npy_dict, dict)

        # Retrieve the two numpy arrays with the column names of the original
        # RDataFrame as dictionary keys.
        npy_x = npy_dict["x"]
        npy_y = npy_dict["y"]
        assert isinstance(npy_x, numpy.ndarray)
        assert isinstance(npy_y, numpy.ndarray)

        # Check the two arrays are of the same length as the original columns.
        assert len(npy_x) == 10
        assert len(npy_y) == 10

        # Check the types correspond to the ones of the original columns.
        int_32_dtype = numpy.dtype("int32")
        float_32_dtype = numpy.dtype("float32")
        assert npy_x.dtype == int_32_dtype
        assert npy_y.dtype == float_32_dtype

    def test_distributed_asnumpy(self, connection):
        """Test support for `AsNumpy` pythonization in distributed backend"""

        # Let's create a simple dataframe with ten rows and two columns
        df = Spark.RDataFrame(10, sparkcontext=connection).Define("x", "(int)rdfentry_")\
            .Define("y", "1.f/(1.f+rdfentry_)")

        # Build a dictionary of numpy arrays.
        npy = df.AsNumpy()
        self.check_npy_dict(npy)

    def test_distributed_asnumpy_columns(self, connection):
        """
        Test that distributed AsNumpy correctly accepts the 'columns' keyword
        argument.
        """

        # Let's create a simple dataframe with ten rows and two columns
        df = Spark.RDataFrame(10, sparkcontext=connection)\
            .Define("x", "(int)rdfentry_")\
            .Define("y", "1.f/(1.f+rdfentry_)")

        # Build a dictionary of numpy arrays.
        npy = df.AsNumpy(columns=["x"])

        # Check the dictionary only has the desired column
        assert list(npy.keys()) == ["x"]

        # Check correctness of the output array
        npy_x = npy["x"]
        assert isinstance(npy_x, numpy.ndarray)
        assert len(npy_x) == 10
        int_32_dtype = numpy.dtype("int32")
        assert npy_x.dtype == int_32_dtype

    def test_distributed_asnumpy_lazy(self, connection):
        """Test that `AsNumpy` can be still called lazily in distributed mode"""

        # Let's create a simple dataframe with ten rows and two columns
        df = Spark.RDataFrame(10, sparkcontext=connection).Define("x", "(int)rdfentry_")\
            .Define("y", "1.f/(1.f+rdfentry_)")

        npy_lazy = df.AsNumpy(lazy=True)
        # The event loop hasn't been triggered yet
        assert isinstance(npy_lazy, ActionProxy)
        assert npy_lazy.proxied_node.value is None

        # Trigger the computations and check final results
        npy = npy_lazy.GetValue()
        self.check_npy_dict(npy)

    def check_snapshot_df(self, snapdf, snapfilename):
        # Count the rows in the snapshotted dataframe
        snapcount = snapdf.Count()

        assert snapcount.GetValue() == 10

        # Retrieve list of file from the snapshotted dataframe
        input_files = snapdf.proxied_node.inputfiles
        # Create list of supposed filenames for the intermediary files
        tmp_files = [f"{snapfilename}_0.root", f"{snapfilename}_1.root"]
        # Check that the two lists are the same
        assert input_files == tmp_files
        # Check that the intermediary .root files were created with the right
        # names, then remove them because they are not necessary
        for filename in tmp_files:
            assert os.path.exists(filename)
            os.remove(filename)

    def test_distributed_snapshot(self, connection):
        """Test support for `Snapshot` in distributed backend"""
        # A simple dataframe with ten sequential numbers from 0 to 9
        df = Spark.RDataFrame(10, sparkcontext=connection).Define("x", "rdfentry_")

        # Snapshot to two files, build a ROOT.TChain with them and retrieve a
        # Spark.RDataFrame
        snapdf = df.Snapshot("snapTree", "snapFile.root")
        self.check_snapshot_df(snapdf, "snapFile")

    def test_distributed_snapshot_columnlist(self, connection):
        """
        Test that distributed Snapshot correctly passes also the third input
        argument "columnList".
        """
        # A simple dataframe with ten sequential numbers from 0 to 9
        df = Spark.RDataFrame(10, sparkcontext=connection)\
            .Define("a", "rdfentry_")\
            .Define("b", "rdfentry_")\
            .Define("c", "rdfentry_")\
            .Define("d", "rdfentry_")

        expectedcolumns = ["a", "b"]
        df.Snapshot("snapTree_columnlist", "snapFile_columnlist.root", expectedcolumns)

        # Create a traditional RDF from the snapshotted files to retrieve the
        # list of columns
        tmp_files = ["snapFile_columnlist_0.root", "snapFile_columnlist_1.root"]
        rdf = ROOT.RDataFrame("snapTree_columnlist", tmp_files)
        snapcolumns = [str(column) for column in rdf.GetColumnNames()]

        assert snapcolumns == expectedcolumns

        for filename in tmp_files:
            os.remove(filename)

    def test_distributed_snapshot_lazy(self, connection):
        """Test that `Snapshot` can be still called lazily in distributed mode"""
        # A simple dataframe with ten sequential numbers from 0 to 9
        df = Spark.RDataFrame(10, sparkcontext=connection).Define("x", "rdfentry_")

        opts = ROOT.RDF.RSnapshotOptions()
        opts.fLazy = True
        snap_lazy = df.Snapshot("snapTree_lazy", "snapFile_lazy.root", ["x"], opts)
        # The event loop hasn't been triggered yet
        assert isinstance(snap_lazy, ActionProxy)
        assert snap_lazy.proxied_node.value is None

        snapdf = snap_lazy.GetValue()
        self.check_snapshot_df(snapdf, "snapFile_lazy")

    def test_redefine_one_column(self, connection):
        """Test that values of one column can be properly redefined."""
        # A simple dataframe with ten sequential numbers from 0 to 9
        df = Spark.RDataFrame(10, sparkcontext=connection)
        df_before = df.Define("x", "1")
        df_after = df_before.Redefine("x", "2")

        # Initial sum should be equal to 10
        sum_before = df_before.Sum("x")
        # Sum after the redefinition should be equal to 20
        sum_after = df_after.Sum("x")

        assert sum_before.GetValue() == 10.0
        assert sum_after.GetValue() == 20.0


if __name__ == "__main__":
    pytest.main(args=[__file__])
