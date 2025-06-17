import unittest
import ROOT
import os

RFile = ROOT.Experimental.RFile

class RFileTests(unittest.TestCase):
    def test_open_for_reading(self):
        """A RFile can read a ROOT file created by TFile"""
        
        fileName = "test_rfile_read_py.root"
    
        # Create a root file to open
        with ROOT.TFile.Open(fileName, "RECREATE") as tfile:
            hist = ROOT.TH1D("hist", "", 100, -10, 10)
            hist.FillRandom("gaus", 100)
            tfile.WriteObject(hist, "hist")

        with RFile.OpenForReading(fileName) as rfile:
            hist = rfile.Get("hist")
            self.assertNotEqual(hist, None)
            self.assertEqual(rfile.Get[ROOT.TH1D]("inexistent"), None)
            self.assertEqual(rfile.Get[ROOT.TH1F]("hist"), None)
            self.assertNotEqual(rfile.Get[ROOT.TH1]("hist"), None)

            with self.assertRaises(ROOT.RException) as _cm:
                rfile.Put("foo", hist)

        os.remove(fileName)


    def test_writing_reading(self):
        """A RFile can be written into and read from"""

        fileName = "test_rfile_writeread_py.root"

        with RFile.Recreate(fileName) as rfile:
            hist = ROOT.TH1D("hist", "", 100, -10, 10)
            hist.FillRandom("gaus", 10)
            rfile.Put("hist", hist)
            with self.assertRaises(ROOT.RException) as _cm:
                rfile.Put("hist/2", hist)

        with RFile.OpenForReading(fileName) as rfile:
            hist = rfile.Get("hist")
            self.assertNotEqual(hist, None)

        os.remove(fileName)
        

if __name__ == "__main__":
    unittest.main()
