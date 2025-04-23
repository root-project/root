import os

import pytest

import ROOT
from DistRDF.Backends import Dask


class TestDaskHistoWrite:
    """
    Integration tests to check writing histograms to a `TFile` distributedly.
    """

    nentries = 10000
    gaus_mean = 10
    gaus_stdev = 1
    delta_equal = 0.01

    def test_write_histo(self, payload):
        """
        Tests that an histogram is correctly written to a .root file created
        before the execution of the event loop.
        """

        # Create a new file where the histogram will be written
        with ROOT.TFile("out_file.root", "recreate") as outfile:
            # We can reuse the same dataset from another test
            treename = "T"
            filename = "../data/ttree/distrdf_roottest_check_friend_trees_main.root"
            # Create a DistRDF RDataFrame with the parent and the friend trees
            connection, _ = payload
            df = ROOT.RDataFrame(treename, filename, executor=connection)

            # Create histogram
            histo = df.Histo1D(("x", "x", 100, 0, 20), "x")

            # Write histogram to out_file.root and close the file
            outfile.WriteObject(histo.GetValue(), histo.GetName())

        # Reopen file to check that histogram was correctly stored
        with ROOT.TFile("out_file.root") as infile:
            reopen_histo = infile.Get("x")
            # Check histogram statistics
            assert reopen_histo.GetEntries() == self.nentries
            assert reopen_histo.GetMean() == pytest.approx(self.gaus_mean, self.delta_equal)
            assert reopen_histo.GetStdDev() == pytest.approx(
                self.gaus_stdev, self.delta_equal)

        # Remove unnecessary .root files
        os.remove("out_file.root")


if __name__ == "__main__":
    pytest.main(args=[__file__])
