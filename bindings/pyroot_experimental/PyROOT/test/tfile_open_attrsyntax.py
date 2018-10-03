import os
import unittest

import ROOT
from libcppyy import SetOwnership


class TFileOpenAndGetattrSyntax(unittest.TestCase):
    """
    Test for the pythonizations of TFile
    """

    filename0  = 'tfilePythonisation_plugin.root'
    filename1  = 'tfilePythonisation.root'
    histname  = 'myHist'
    nbins = 128
    minx = -4
    maxx = 4

    # Setup
    @classmethod
    def setUpClass(cls):
       f0 = ROOT.TFile.Open(cls.filename0, "RECREATE")
       h = ROOT.TH1F(cls.histname, cls.histname, cls.nbins, cls.minx, cls.maxx)
       SetOwnership(h, False)
       h.Write()
       f0.Close()

       h = ROOT.TH1F(cls.histname, cls.histname, cls.nbins, cls.minx, cls.maxx)
       f1 = ROOT.TFile.Open(cls.filename1, "RECREATE")
       f1.WriteObject(h, cls.histname)
       f1.Close()

    # Helpers
    def check_histo(self, h):
       self.assertEqual(h.GetName(), self.histname)
       self.assertEqual(h.GetTitle(), self.histname)
       self.assertEqual(h.GetNbinsX(), self.nbins)
       xaxis = h.GetXaxis()
       self.assertEqual(xaxis.GetXmin(), self.minx)
       self.assertEqual(xaxis.GetXmax(), self.maxx)

    def get_histo_and_check(self, f):
        h = f.myHist
        self.check_histo(h)
        hh = f.Get("myHist")
        self.check_histo(hh)

    # Tests
    def test_read0(self):
        f0 = ROOT.TFile.Open(self.filename0)
        self.get_histo_and_check(f0)

    def test_read1(self):
        f1 = ROOT.TFile(self.filename1)
        self.get_histo_and_check(f1)

if __name__ == '__main__':
    unittest.main()
