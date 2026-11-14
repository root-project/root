import unittest

import ROOT

RBinIndex = ROOT.Experimental.RBinIndex
RHist = ROOT.Experimental.RHist
RHistEngine = ROOT.Experimental.RHistEngine
RRegularAxis = ROOT.Experimental.RRegularAxis


class HistEngine_Indexing(unittest.TestCase):
    def test_access(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["int"](axis)

        # Fill and access a normal bin
        engine.Fill(1.5)
        self.assertEqual(engine[1], 1)
        self.assertEqual(engine[RBinIndex(1)], 1)

        with self.assertRaises(IndexError):
            engine[Bins]

        with self.assertRaises(IndexError):
            engine[0, 1]

        # Fill and access the underflow bin
        engine.Fill(-100)
        self.assertEqual(engine[ROOT.uhi.underflow], 1)
        self.assertEqual(engine[RBinIndex.Underflow()], 1)

        # Fill and access the overflow bin
        engine.Fill(100)
        self.assertEqual(engine[ROOT.uhi.overflow], 1)
        self.assertEqual(engine[RBinIndex.Overflow()], 1)

        # Test multidimensional histogram
        engine = RHistEngine["int"](axis, axis)
        engine.Fill(1.5, 2.5)
        self.assertEqual(engine[1, 2], 1)
        self.assertEqual(engine[RBinIndex(1), RBinIndex(2)], 1)

        with self.assertRaises(IndexError):
            engine[0]

    def test_slicing(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["int"](axis)

        # Slice without operation
        self.assertEqual(engine[:].GetTotalNBins(), 22)
        self.assertEqual(engine[3:11].GetTotalNBins(), 10)
        self.assertEqual(engine[:11].GetTotalNBins(), 13)
        self.assertEqual(engine[3:].GetTotalNBins(), 19)
        self.assertEqual(engine[0:len].GetTotalNBins(), 22)

        # The implementation requires normal bins
        with self.assertRaises(IndexError):
            engine[ROOT.uhi.underflow :]

        # This doesn't even make sense because the endpoint is exclusive...
        with self.assertRaises(IndexError):
            engine[: ROOT.uhi.overflow]

        # Rebin operation
        self.assertEqual(engine[:: ROOT.uhi.rebin(2)].GetTotalNBins(), 12)
        self.assertEqual(engine[3 : 11 : ROOT.uhi.rebin(2)].GetTotalNBins(), 6)
        # Syntactic sugar without an actual slice
        self.assertEqual(engine[ROOT.uhi.rebin(2)].GetTotalNBins(), 12)

        with self.assertRaises(IndexError):
            engine[:, :]

        # Test multidimensional slicing
        engine = RHistEngine["int"](axis, axis)
        self.assertEqual(engine[:, :].GetNDimensions(), 2)

        with self.assertRaises(IndexError):
            engine[:]

        # Sum operation
        self.assertEqual(engine[:, :: ROOT.uhi.sum].GetNDimensions(), 1)
        self.assertEqual(engine[:, 3 : 11 : ROOT.uhi.sum].GetNDimensions(), 1)
        self.assertEqual(engine[:, 0 : len : ROOT.uhi.sum].GetNDimensions(), 1)

        # The implementation requires either none or both endpoints
        with self.assertRaises(IndexError):
            engine[:, 0::sum]

        with self.assertRaises(IndexError):
            engine[:, :2:sum]

        # A single value v is like v:v+1:sum
        self.assertEqual(engine[:, 2].GetNDimensions(), 1)

        with self.assertRaises(IndexError):
            engine[:, Bins]

        with self.assertRaises(IndexError):
            engine[:, ROOT.uhi.underflow]

        # Syntactic sugar: builtin sum, and / or without an actual slice
        self.assertEqual(engine[:, ::sum].GetNDimensions(), 1)
        self.assertEqual(engine[:, ROOT.uhi.sum].GetNDimensions(), 1)
        self.assertEqual(engine[:, sum].GetNDimensions(), 1)

    def test_slicing_expand(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["int"](axis, axis)

        # Ellipsis
        self.assertEqual(engine[...].GetNDimensions(), 2)

        sliced = engine[3:11, ...]
        self.assertEqual(sliced.GetNDimensions(), 2)
        self.assertEqual(sliced.GetAxes()[0].GetNNormalBins(), 8)

        sliced = engine[..., 3:11]
        self.assertEqual(sliced.GetNDimensions(), 2)
        self.assertEqual(sliced.GetAxes()[1].GetNNormalBins(), 8)

        with self.assertRaises(IndexError):
            engine[..., ...]

        # Dictionary
        self.assertEqual(engine[{}].GetNDimensions(), 2)
        self.assertEqual(engine[{0: slice(None)}].GetNDimensions(), 2)
        self.assertEqual(engine[{0: sum}].GetNDimensions(), 1)

        with self.assertRaises(IndexError):
            engine[{0.0: sum}]

        with self.assertRaises(IndexError):
            engine[{2: slice(None)}]

    def test_setting(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["int"](axis)

        # Set a normal bin
        engine[1] = 2
        self.assertEqual(engine.GetBinContent(RBinIndex(1)), 2)

        with self.assertRaises(IndexError):
            engine[Bins] = 2

        with self.assertRaises(IndexError):
            engine[0, 1] = 2

        # Set the underflow bin
        engine[ROOT.uhi.underflow] = 3
        self.assertEqual(engine.GetBinContent(RBinIndex.Underflow()), 3)

        # Set the overflow bin
        engine[ROOT.uhi.overflow] = 4
        self.assertEqual(engine.GetBinContent(RBinIndex.Overflow()), 4)

        # Test multidimensional histogram
        engine = RHistEngine["int"](axis, axis)
        engine[1, 2] = 5
        self.assertEqual(engine.GetBinContent(RBinIndex(1), RBinIndex(2)), 5)

        with self.assertRaises(IndexError):
            engine[0] = 5


class Hist_Indexing(unittest.TestCase):
    def test_access(self):
        Bins = 20
        hist = RHist["int"](Bins, (0, Bins))

        # Fill and access a normal bin
        hist.Fill(1.5)
        self.assertEqual(hist[1], 1)
        self.assertEqual(hist[RBinIndex(1)], 1)

        with self.assertRaises(IndexError):
            hist[Bins]

        with self.assertRaises(IndexError):
            hist[0, 1]

    def test_slicing(self):
        Bins = 20
        hist = RHist["int"](Bins, (0, Bins))

        self.assertEqual(hist[:].GetTotalNBins(), 22)

        with self.assertRaises(IndexError):
            hist[:, :]

    def test_setting(self):
        Bins = 20
        hist = RHist["int"](Bins, (0, Bins))
        self.assertFalse(hist.GetStats().IsTainted())

        # Set a normal bin
        hist[1] = 2
        self.assertEqual(hist.GetBinContent(RBinIndex(1)), 2)
        self.assertTrue(hist.GetStats().IsTainted())

        with self.assertRaises(IndexError):
            hist[Bins] = 2

        with self.assertRaises(IndexError):
            hist[0, 1] = 2
