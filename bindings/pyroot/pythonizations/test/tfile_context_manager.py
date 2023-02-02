import os
import unittest

import ROOT

from ROOT import TFile


class TFileContextManager(unittest.TestCase):
    """
    Test of TFile used as context manager
    """

    NBINS = 123
    XMIN = 10
    XMAX = 242

    def check_file_data(self, tfile, filename):
        """
        Check status of the TFile after the context manager and correctness of
        the data it contains.
        """
        self.assertTrue(tfile)  # The TFile object is still there
        self.assertFalse(tfile.IsOpen())  # And it is correctly closed

        with TFile(filename, "read") as infile:
            hin = infile.Get("myhisto")
            xaxis = hin.GetXaxis()
            self.assertEqual(self.NBINS, hin.GetNbinsX())
            self.assertEqual(self.XMIN, xaxis.GetXmin())
            self.assertEqual(self.XMAX, xaxis.GetXmax())

        os.remove(filename)

    def test_writeobject(self):
        """
        Write a histogram in a file within a context manager, using TDirectory::WriteObject.
        """
        filename = "TFileContextManager_test_writeobject.root"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F("myhisto", "myhisto", self.NBINS, self.XMIN, self.XMAX)
            outfile.WriteObject(hout, "myhisto")

        self.check_file_data(outfile, filename)

    def test_histowrite(self):
        """
        Write a histogram in a file within a context manager, using TH1::Write.
        """
        filename = "TFileContextManager_test_histowrite.root"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F("myhisto", "mhisto", self.NBINS, self.XMIN, self.XMAX)
            hout.Write()

        self.check_file_data(outfile, filename)

    def test_filewrite(self):
        """
        Write a histogram in a file within a context manager, using TFile::Write.
        """
        filename = "TFileContextManager_test_filewrite.root"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F("myhisto", "myhisto", self.NBINS, self.XMIN, self.XMAX)
            outfile.Write()

        self.check_file_data(outfile, filename)

    def test_detachhisto(self):
        """
        Detach histogram from file and access it outside of the context, both when writing and reading.
        """
        filename = "TFileContextManager_test_detachhisto.root"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F("myhisto", "myhisto", self.NBINS, self.XMIN, self.XMAX)
            hout.SetDirectory(ROOT.nullptr)
            outfile.WriteObject(hout, "myhisto")

        self.assertTrue(hout)
        self.assertEqual(hout.GetName(), "myhisto")

        with TFile(filename, "read") as infile:
            hin = infile.Get("myhisto")
            hin.SetDirectory(ROOT.nullptr)
            xaxis = hin.GetXaxis()
            self.assertEqual(self.NBINS, hin.GetNbinsX())
            self.assertEqual(self.XMIN, xaxis.GetXmin())
            self.assertEqual(self.XMAX, xaxis.GetXmax())

        self.assertTrue(hin)
        self.assertEqual(hin.GetName(), "myhisto")

        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
