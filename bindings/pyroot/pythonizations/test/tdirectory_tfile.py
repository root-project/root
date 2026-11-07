import os
import unittest

import ROOT
from ROOT import TFile


class TDirectoryReadWrite(unittest.TestCase):
    """
    Test for the getitem syntax of TDirectory.
    """

    nbins = 8
    xmin = 0
    xmax = 4

    # Setup
    @classmethod
    def setUpClass(cls):
        cls.dir0 = ROOT.TDirectory("dir0", "dir0")
        h = ROOT.TH1F("h", "h", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h, False)
        # this must be there otherwise the histogram is not attached to dir0
        h.SetDirectory(cls.dir0)

        dir1 = cls.dir0.mkdir("dir1")
        dir1.cd()
        h1 = ROOT.TH1F("h1", "h1", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h1, False)
        h1.SetDirectory(dir1)

        dir2 = dir1.mkdir("dir2")
        dir2.cd()
        h2 = ROOT.TH1F("h2", "h2", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h2, False)
        h2.SetDirectory(dir2)

    @classmethod
    def tearDownClass(cls):
        # Release the directory (and the histograms it owns) now, while ROOT is
        # fully initialised, instead of leaving it to interpreter shutdown,
        # where we risk colliding with other directory objects of the same name
        # from other tests.
        cls.dir0 = None

    def checkHisto(self, h):
        xaxis = h.GetXaxis()
        self.assertEqual(self.nbins, h.GetNbinsX())
        self.assertEqual(self.xmin, xaxis.GetXmin())
        self.assertEqual(self.xmax, xaxis.GetXmax())

    # Tests
    def test_readHisto_itemsyntax(self):
        self.checkHisto(self.dir0["h"])
        self.checkHisto(self.dir0["dir1"]["h1"])
        self.checkHisto(self.dir0["dir1"]["dir2"]["h2"])

    def test_caching_getitem(self):
        # check that object is not cached initially
        self.assertFalse("h" in self.dir0.__dict__)
        self.dir0["h"]
        # check that the cached value in is actually the object
        # inside the directory
        self.assertTrue(self.dir0._cached_items["h"] is self.dir0["h"])


class TDirectoryFileReadWrite(unittest.TestCase):
    """
    Test for the getitem syntax and Get method of TDirectoryFile.
    """

    nbins = 8
    xmin = 0
    xmax = 4

    # Setup
    @classmethod
    def setUpClass(cls):
        cls.dir0 = ROOT.TDirectoryFile("dir0", "dir0")
        h = ROOT.TH1F("h", "h", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h, False)
        # this must be there otherwise the histogram is not attached to dir0
        h.SetDirectory(cls.dir0)

        dir1 = cls.dir0.mkdir("dir1")
        dir1.cd()
        h1 = ROOT.TH1F("h1", "h1", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h1, False)
        h1.SetDirectory(dir1)

        dir2 = dir1.mkdir("dir2")
        dir2.cd()
        h2 = ROOT.TH1F("h2", "h2", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h2, False)
        h2.SetDirectory(dir2)

    @classmethod
    def tearDownClass(cls):
        # Release the directory (and the histograms it owns) now, while ROOT is
        # fully initialised, instead of leaving it to interpreter shutdown,
        # where we risk colliding with other directory objects of the same name
        # from other tests.
        cls.dir0 = None

    def checkHisto(self, h):
        xaxis = h.GetXaxis()
        self.assertEqual(self.nbins, h.GetNbinsX())
        self.assertEqual(self.xmin, xaxis.GetXmin())
        self.assertEqual(self.xmax, xaxis.GetXmax())

    # Tests
    def test_readHisto_itemsyntax(self):
        self.checkHisto(self.dir0["h"])
        self.checkHisto(self.dir0["dir1"]["h1"])
        self.checkHisto(self.dir0["dir1"]["dir2"]["h2"])

    def test_readHisto(self):
        self.checkHisto(self.dir0.Get("h"))
        self.checkHisto(self.dir0.Get("dir1/h1"))
        self.checkHisto(self.dir0.Get("dir1/dir2/h2"))

    def test_caching_getitem(self):
        # check that object is not cached initially
        self.assertFalse(hasattr(self.dir0, "_cached_items"))
        self.dir0["h"]
        # check that the value in __dict__ is actually the object
        # inside the directory
        self.assertTrue(self.dir0._cached_items['h'] is self.dir0["h"])


class TFileOpenReadWrite(unittest.TestCase):
    """
    Test for the TFile.Open factory like creation of TFile
    """

    filename  = 'tfileopenreadwrite.root'
    nbins = 8
    xmin = 0
    xmax = 4

    # Setup
    @classmethod
    def setUpClass(cls):
        f = ROOT.TFile.Open(cls.filename, "RECREATE")
        h = ROOT.TH1F("h", "h", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h, False)
        f.WriteObject(h, "h")

        dir1 = f.mkdir("dir1")
        dir1.cd()
        h1 = ROOT.TH1F("h1", "h1", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h1, False)
        h1.Write()

        dir2 = dir1.mkdir("dir2")
        dir2.cd()
        h2 = ROOT.TH1F("h2", "h2", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h2, False)
        h2.Write()

        f.Close()

    def checkHisto(self, h):
        xaxis = h.GetXaxis()
        self.assertEqual(self.nbins, h.GetNbinsX())
        self.assertEqual(self.xmin, xaxis.GetXmin())
        self.assertEqual(self.xmax, xaxis.GetXmax())

    # Tests
    def test_readHisto_itemsyntax(self):
        f = ROOT.TFile.Open(self.filename)
        self.checkHisto(f["h"])
        self.checkHisto(f["dir1"]["h1"])
        self.checkHisto(f["dir1"]["dir2"]["h2"])

    def test_readHisto(self):
        f = ROOT.TFile.Open(self.filename)
        self.checkHisto(f.Get("h"))
        self.checkHisto(f.Get("dir1/h1"))
        self.checkHisto(f.Get("dir1/dir2/h2"))

    def test_caching_getitem(self):
        f = ROOT.TFile.Open(self.filename)
        # check that object is not cached initially
        self.assertFalse(hasattr(f, "_cached_items"))
        f["h"]
        # check that the value in __dict__ is actually the object
        # inside the directory
        self.assertTrue(f._cached_items['h'] is f["h"])

    def test_oserror(self):
        # check that an OSError is raised when an inexistent file is opened
        # both with a string and an instance of TFileOpenHandle as arguments
        self.assertRaises(OSError, ROOT.TFile.Open, 'inexistent_file.root')
        handle = ROOT.TFile.AsyncOpen("inexistent_file.root")
        self.assertRaises(OSError, ROOT.TFile.Open, handle)

    def test_keys_title(self):
        """
        Test that the TKey related to a histogram in the file contains the
        histogram title as described in #9989.
        """
        finput = ROOT.TFile.Open(self.filename)
        key1 = finput.GetListOfKeys().At(0)
        key2 = finput.Get("dir1").GetListOfKeys().At(0)
        key3 = finput.Get("dir1/dir2").GetListOfKeys().At(0)
        self.assertEqual(key1.GetTitle(), "h")
        self.assertEqual(key2.GetTitle(), "h1")
        self.assertEqual(key3.GetTitle(), "h2")
        finput.Close()


class TFileConstructor(unittest.TestCase):
    """
    Test for the TFile constructor
    """

    def test_oserror(self):
        # check that an OSError is raised when the string passed as argument
        # refers to an inexistent file
        self.assertRaises(OSError, ROOT.TFile, 'inexistent_file.root')


class TFileContextManager(unittest.TestCase):
    """
    Test of TFile used as context manager
    """

    NBINS = 123
    XMIN = 10
    XMAX = 242

    def check_file_data(self, tfile, filename, histoname):
        """
        Check status of the TFile after the context manager and correctness of
        the data it contains.
        """
        self.assertTrue(tfile)  # The TFile object is still there
        self.assertFalse(tfile.IsOpen())  # And it is correctly closed

        with TFile(filename, "read") as infile:
            hin = infile.Get(histoname)
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
        histoname = "myhisto"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F(histoname, histoname, self.NBINS, self.XMIN, self.XMAX)
            outfile.WriteObject(hout, "myhisto")

        self.check_file_data(outfile, filename, histoname)

    def test_histowrite(self):
        """
        Write a histogram in a file within a context manager, using TH1::Write.
        """
        filename = "TFileContextManager_test_histowrite.root"
        histoname = "myhisto_2"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F(histoname, histoname, self.NBINS, self.XMIN, self.XMAX)
            hout.Write()

        self.check_file_data(outfile, filename, histoname)

    def test_filewrite(self):
        """
        Write a histogram in a file within a context manager, using TFile::Write.
        """
        filename = "TFileContextManager_test_filewrite.root"
        histoname = "myhisto_3"
        with TFile(filename, "recreate") as outfile:
            hout = ROOT.TH1F(histoname, histoname, self.NBINS, self.XMIN, self.XMAX)
            hout.SetDirectory(outfile)
            outfile.Write()

        self.check_file_data(outfile, filename, histoname)

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
