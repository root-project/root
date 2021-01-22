import PyRDF
import unittest
import os


class RDataFrameSnapshot(unittest.TestCase):
    """Test `Snapshot` functionality for RDataFrame"""

    def test_snapshot_nrows(self):
        """Test support for `Snapshot` in local backend"""
        def fill_tree(treeName, fileName):
            rdf = PyRDF.RDataFrame(100)
            return rdf.Define("b1", "rdfentry_")\
                      .Snapshot(treeName, fileName)

        # We prepare an input tree to run on
        fileName = "snapFile.root"
        treeName = "snapTree"

        snapdf = fill_tree(treeName, fileName)

        # We read the tree from the file and create a RDataFrame.
        d = PyRDF.RDataFrame(treeName, fileName)

        # Check on dataframe retrieved from file
        d_cut = d.Filter("b1 % 2 == 0")

        d_count = d_cut.Count()

        self.assertEqual(d_count.GetValue(), 50)

        # Check on dataframe returned by Snapshot operation
        snapdf_cut = snapdf.Filter("b1 % 2 == 0")
        snapdf_count = snapdf_cut.Count()

        self.assertEqual(snapdf_count.GetValue(), 50)

        # Remove unnecessary .root file
        os.remove(fileName)
