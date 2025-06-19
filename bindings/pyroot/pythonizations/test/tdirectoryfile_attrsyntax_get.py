import unittest

import ROOT


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

        dir2 = dir1.mkdir("dir2")
        dir2.cd()
        h2 = ROOT.TH1F("h2", "h2", cls.nbins, cls.xmin, cls.xmax)
        ROOT.SetOwnership(h2, False)

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


if __name__ == '__main__':
    unittest.main()
