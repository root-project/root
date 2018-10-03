import os
import unittest

import ROOT
from libcppyy import SetOwnership


class TDirectoryWriteAndGetObject(unittest.TestCase):
    """
    Test the WriteObject method of TDirectory
    """
    filename = 'TDirectoryWriteAndGetObject.root'
    dirName  = 'myDir'
    directory = None
    histName  = 'myHist'
    hkeyName = 'myKey'
    nbins = 128
    minx = -4
    maxx = 4
    histo = None
    stKeyName = "stKey"
    stm1 = 42
    stm2 = -7

    # Setup
    @classmethod
    def setUpClass(self):
       self.histo = ROOT.TH1F(self.histName, self.histName, self.nbins, self.minx, self.maxx)
       self.directory = ROOT.gDirectory.mkdir(self.dirName)
       self.directory.Append(self.histo)

       ROOT.gInterpreter.ProcessLine(".L mystruct.h+")
       # In order to properly test the features, we attach the directory to
       # a real file
       f = ROOT.TFile(self.filename, "RECREATE")
       d = f.mkdir(self.dirName)
       st = ROOT.MyStruct()
       st.myint1 = self.stm1
       st.myint2 = self.stm2
       d.WriteObject(st, self.stKeyName)
       f.Close()

    # Helpers
    def check_histo(self, h):
       self.assertEqual(h.GetName(), self.histName)
       self.assertEqual(h.GetTitle(), self.histName)
       self.assertEqual(h.GetNbinsX(), self.nbins)
       xaxis = h.GetXaxis()
       self.assertEqual(xaxis.GetXmin(), self.minx)
       self.assertEqual(xaxis.GetXmax(), self.maxx)

    def check_st(self, st):
       self.assertEqual(st.myint1, self.stm1)
       self.assertEqual(st.myint2, self.stm2)

    # Tests
    def test_histo_directory_get(self):
       h = self.directory.Get(self.histName)
       self.check_histo(h)

    def test_histo_directory_attrsyntax(self):
       h = self.directory.myHist
       self.check_histo(h)

    def test_obj_file_get(self):
       f = ROOT.TFile(self.filename)
       d = f.Get(self.dirName)
       st = d.Get(self.stKeyName)
       self.check_st(st)

    def test_obj_file_attrsyntax(self):
       f = ROOT.TFile(self.filename)
       st = f.myDir.stKey
       self.check_st(st)

if __name__ == '__main__':
    unittest.main()
