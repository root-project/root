import os
import unittest
from collections import namedtuple

import ROOT

from DistRDF.Backends import Dask

from dask.distributed import Client, LocalCluster


class DaskHistogramsTest(unittest.TestCase):
    """Integration tests to check the working of DistRDF."""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        - Disable ROOT graphics.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

        cls.oldbatch = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(True)

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()
        cls.client.close()

        ROOT.gROOT.SetBatch(cls.oldbatch)

    def build_distrdf_graph(self):
        """
        Create a DistRDF graph with a fixed set of operations and return it.
        """
        treename = "data"
        files = ["http://root.cern/files/teaching/CMS_Open_Dataset.root", ]
        rdf = Dask.RDataFrame(treename, files, npartitions=5, daskclient=self.client)

        # Define the analysis cuts
        chargeCutStr = "C1 != C2"
        etaCutStr = "fabs(eta1) < 2.3 && fabs(eta2) < 2.3"
        ptCutStr = "pt1 > 2 && pt2 > 2"
        rdf_f = rdf.Filter(chargeCutStr, "Opposite Charge") \
                   .Filter(etaCutStr, "Central Muons") \
                   .Filter(ptCutStr, "Sane Pt")

        # Create the invariant mass column
        invMassFormulaStr = ("sqrt(pow(E1+E2, 2) - (pow(px1+px2, 2) +"
                             "pow(py1+py2, 2) + pow(pz1+pz2, 2)))")
        rdf_fd = rdf_f.Define("invMass", invMassFormulaStr)

        # Create the histograms
        pt1_h = rdf.Histo1D(("", "", 128, 1, 1200), "pt1")
        pt2_h = rdf.Histo1D(("", "", 128, 1, 1200), "pt2")
        model = (
            "invMass", "CMS Opendata;#mu#mu mass[GeV];Events", 512, 5, 110)
        invMass_h = rdf_fd.Histo1D(model, "invMass")
        pi = ROOT.TMath.Pi()
        model = ("", "", 64, -pi, pi, 64, -pi, pi)
        phis_h = rdf_fd.Histo2D(model, "phi1", "phi2")

        return pt1_h, pt2_h, invMass_h, phis_h

    def build_rootrdf_graph(self):
        """Create an RDF graph with a fixed set of operations and return it."""
        treename = "data"
        files = ["http://root.cern/files/teaching/CMS_Open_Dataset.root", ]
        rdf = ROOT.RDataFrame(treename, files)

        # Define the analysis cuts
        chargeCutStr = "C1 != C2"
        etaCutStr = "fabs(eta1) < 2.3 && fabs(eta2) < 2.3"
        ptCutStr = "pt1 > 2 && pt2 > 2"
        rdf_f = rdf.Filter(chargeCutStr, "Opposite Charge") \
                   .Filter(etaCutStr, "Central Muons") \
                   .Filter(ptCutStr, "Sane Pt")

        # Create the invariant mass column
        invMassFormulaStr = ("sqrt(pow(E1+E2, 2) - (pow(px1+px2, 2) +"
                             "pow(py1+py2, 2) + pow(pz1+pz2, 2)))")
        rdf_fd = rdf_f.Define("invMass", invMassFormulaStr)

        # Create the histograms
        pt1_h = rdf.Histo1D(("", "", 128, 1, 1200), "pt1")
        pt2_h = rdf.Histo1D(("", "", 128, 1, 1200), "pt2")
        model = (
            "invMass", "CMS Opendata;#mu#mu mass[GeV];Events", 512, 5, 110)
        invMass_h = rdf_fd.Histo1D(model, "invMass")
        pi = ROOT.TMath.Pi()
        model = ("", "", 64, -pi, pi, 64, -pi, pi)
        phis_h = rdf_fd.Histo2D(model, "phi1", "phi2")

        return pt1_h, pt2_h, invMass_h, phis_h

    def test_dask_histograms(self):
        """Check that Dask backend works the same way as ROOT RDF."""
        physics_variables = ["pt1_h", "pt2_h", "invMass_h", "phis_h"]

        DaskResult = namedtuple("DaskResult", physics_variables)
        daskresults = DaskResult(*self.build_distrdf_graph())

        daskresults.pt1_h.Draw("PL PLC PMC")  # Trigger Event-loop, Dask

        # ROOT RDataFrame execution
        RDFResult = namedtuple("RDFResult", physics_variables)
        rootrdf = RDFResult(*self.build_rootrdf_graph())

        rootrdf.pt1_h.Draw("PL PLC PMC")  # Trigger Event-loop, RDF-only

        # Assert 'pt1_h' histogram
        self.assertEqual(daskresults.pt1_h.GetEntries(), rootrdf.pt1_h.GetEntries())
        # Assert 'pt2_h' histogram
        self.assertEqual(daskresults.pt2_h.GetEntries(), rootrdf.pt2_h.GetEntries())
        # Assert 'invMass_h' histogram
        self.assertEqual(daskresults.invMass_h.GetEntries(),
                         rootrdf.invMass_h.GetEntries())
        # Assert 'phis_h' histogram
        self.assertEqual(daskresults.phis_h.GetEntries(),
                         rootrdf.phis_h.GetEntries())


if __name__ == "__main__":
    unittest.main(argv=[__file__])
