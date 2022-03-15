import os

from array import array

import pytest

import ROOT
from DistRDF.Backends import Spark


@pytest.fixture(scope="class")
def setup_testhistowrite(request):
    """
    Set up test environment for this class. Currently this includes:

    - Class-wide histogram-related parameters.
    """

    request.cls.nentries = 10000  # Number of fills
    request.cls.gaus_mean = 10  # Mean of the gaussian distribution
    request.cls.gaus_stdev = 1  # Standard deviation of the gaussian distribution
    request.cls.delta_equal = 0.01  # Delta to check for float equality


@pytest.mark.usefixtures("setup_testhistowrite")
class TestSparkHistoWrite:
    """
    Integration tests to check writing histograms to a `TFile` distributedly.
    """

    def create_tree_with_data(self):
        """Creates a .root file with some data"""
        f = ROOT.TFile("tree_gaus.root", "recreate")
        T = ROOT.TTree("Events", "Gaus(10,1)")

        x = array("f", [0])
        T.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The parent will have a gaussian distribution with mean 10 and
        # standard deviation 1
        for i in range(self.nentries):
            x[0] = r.Gaus(self.gaus_mean, self.gaus_stdev)
            T.Fill()

        f.Write()
        f.Close()

    def test_write_histo(self, connection):
        """
        Tests that an histogram is correctly written to a .root file created
        before the execution of the event loop.
        """
        self.create_tree_with_data()

        # Create a new file where the histogram will be written
        outfile = ROOT.TFile("out_file.root", "recreate")

        # Create a DistRDF RDataFrame with the parent and the friend trees
        df = Spark.RDataFrame("Events", "tree_gaus.root", sparkcontext=connection)

        # Create histogram
        histo = df.Histo1D(("x", "x", 100, 0, 20), "x")

        # Write histogram to out_file.root and close the file
        histo.Write()
        outfile.Close()

        # Reopen file to check that histogram was correctly stored
        reopen_file = ROOT.TFile("out_file.root", "read")
        reopen_histo = reopen_file.Get("x")

        # Check histogram statistics
        assert reopen_histo.GetEntries() == self.nentries
        assert reopen_histo.GetMean() == pytest.approx(self.gaus_mean, self.delta_equal)
        assert reopen_histo.GetStdDev() == pytest.approx(self.gaus_stdev, self.delta_equal)

        # Remove unnecessary .root files
        os.remove("tree_gaus.root")
        os.remove("out_file.root")


if __name__ == "__main__":
    pytest.main(args=[__file__])
