import os
import unittest

import ROOT
from libcppyy import SetOwnership


class TDirectoryUnits(unittest.TestCase):
    """
    Test for the pythonizations of TFile
    """

    dirName  = 'myDir'
    dirName2  = 'myDir2'
    directory = None
    histName  = 'myHist'
    keyName = 'myKey'
    fileName  = 'tdirectoryWrite.root'
    hist = None
    nbins = 128
    minx = -4
    maxx = 4

    # Setup
    @classmethod
    def setUpClass(cls):
       cls.directory = ROOT.gDirectory.mkdir(cls.dirName)
       cls.hist = ROOT.TH1F(cls.histName, cls.histName, cls.nbins, cls.minx, cls.maxx)
       cls.hist.SetDirectory(cls.directory)
       SetOwnership(cls.hist, False) # Is this a bug in cppyy?

       f = ROOT.TFile(cls.fileName,"RECREATE")
       d = f.mkdir(cls.dirName2)
       d.cd()
       ROOT.TH1.AddDirectory(0)
       h = ROOT.TH1F(cls.histName, cls.histName, cls.nbins, cls.minx, cls.maxx)
       d.WriteObject(h, cls.keyName)
       f.Close()

    # Helpers
    def check_histo(self, h):
       self.assertEqual(h.GetName(), self.histName)
       self.assertEqual(h.GetTitle(), self.histName)
       self.assertEqual(h.GetNbinsX(), self.nbins)
       xaxis = h.GetXaxis()
       self.assertEqual(xaxis.GetXmin(), self.minx)
       self.assertEqual(xaxis.GetXmax(), self.maxx)

    # Tests
    def test_get_object(self):
        h = ROOT.TH1F()
        self.directory.GetObject(self.histName, h)
        self.check_histo(h)

    # Tests
    def test_write_object(self):
       f = ROOT.TFile.Open(self.fileName)
       d = getattr(f, self.dirName2)
       hh = ROOT.TH1F()
       d.GetObject(self.keyName, hh)
       self.check_histo(hh)


if __name__ == '__main__':
    unittest.main()
