import os
import unittest

import ROOT
from libcppyy import SetOwnership


class TDirectoryUnits(unittest.TestCase):
    """
    Test for the pythonizations of TFile
    """

    dirName  = 'myDir'
    directory = None
    histname  = 'myHist'
    hist = None
    nbins = 128
    minx = -4
    maxx = 4

    # Setup
    @classmethod
    def setUpClass(cls):
       cls.directory = ROOT.gDirectory.mkdir(cls.dirName)
       cls.hist = ROOT.TH1F(cls.histname, cls.histname, cls.nbins, cls.minx, cls.maxx)
       cls.hist.SetDirectory(cls.directory)
       SetOwnership(cls.hist, False) # Is this a bug in cppyy?

    # Helpers
    def check_histo(self, h):
       self.assertEqual(h.GetName(), self.histname)
       self.assertEqual(h.GetTitle(), self.histname)
       self.assertEqual(h.GetNbinsX(), self.nbins)
       xaxis = h.GetXaxis()
       self.assertEqual(xaxis.GetXmin(), self.minx)
       self.assertEqual(xaxis.GetXmax(), self.maxx)

    # Tests
    def test_get_object(self):
        h = ROOT.TH1F()
        self.directory.ls()
        self.directory.GetObject(self.histname, h)
        self.check_histo(h)

if __name__ == '__main__':
    unittest.main()
