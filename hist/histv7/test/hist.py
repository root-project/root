import math
import unittest

import ROOT

RAxisVariant = ROOT.Experimental.RAxisVariant
RBinIndex = ROOT.Experimental.RBinIndex
RCategoricalAxis = ROOT.Experimental.RCategoricalAxis
RHist = ROOT.Experimental.RHist
RHistEngine = ROOT.Experimental.RHistEngine
RHistStats = ROOT.Experimental.RHistStats
RRegularAxis = ROOT.Experimental.RRegularAxis
RVariableBinAxis = ROOT.Experimental.RVariableBinAxis
RWeight = ROOT.Experimental.RWeight


class RegularAxis(unittest.TestCase):
    def test_init(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins + 2)
        self.assertEqual(axis.GetLow(), 0)
        self.assertEqual(axis.GetHigh(), Bins)
        self.assertTrue(axis.HasFlowBins())

        axis = RRegularAxis(Bins, (0, Bins), enableFlowBins=False)
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins)
        self.assertFalse(axis.HasFlowBins())

    def test_GetNormalRange(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        self.assertEqual(len(list(axis.GetNormalRange())), Bins)
        self.assertEqual(len(list(axis.GetNormalRange(1, 5))), 4)

    def test_GetFullRange(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        self.assertEqual(len(list(axis.GetFullRange())), Bins + 2)


class VariableBinAxis(unittest.TestCase):
    def test_init(self):
        Bins = 20
        # list of int is automatically converted to std::vector<double>.
        bins = [i for i in range(0, Bins + 1)]
        axis = RVariableBinAxis(bins)
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins + 2)
        self.assertEqual(axis.GetBinEdges()[0], 0)
        self.assertEqual(axis.GetBinEdges()[-1], Bins)
        self.assertTrue(axis.HasFlowBins())

        # list of float is also supported.
        bins = [i + 0.5 for i in range(0, Bins + 1)]
        axis = RVariableBinAxis(bins)
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins + 2)
        self.assertEqual(axis.GetBinEdges()[0], 0.5)
        self.assertEqual(axis.GetBinEdges()[-1], Bins + 0.5)
        self.assertTrue(axis.HasFlowBins())

        axis = RVariableBinAxis(bins, enableFlowBins=False)
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins)
        self.assertFalse(axis.HasFlowBins())

    def test_GetNormalRange(self):
        Bins = 20
        bins = [i for i in range(0, Bins + 1)]
        axis = RVariableBinAxis(bins)
        self.assertEqual(len(list(axis.GetNormalRange())), Bins)
        self.assertEqual(len(list(axis.GetNormalRange(1, 5))), 4)

    def test_GetFullRange(self):
        Bins = 20
        bins = [i for i in range(0, Bins + 1)]
        axis = RVariableBinAxis(bins)
        self.assertEqual(len(list(axis.GetFullRange())), Bins + 2)


class CategoricalAxis(unittest.TestCase):
    def test_init(self):
        categories = ["a", "b", "c"]
        axis = RCategoricalAxis(categories)
        self.assertEqual(axis.GetNNormalBins(), 3)
        self.assertEqual(axis.GetTotalNBins(), 4)
        self.assertTrue(axis.HasOverflowBin())

        axis = RCategoricalAxis(categories, enableOverflowBin=False)
        self.assertEqual(axis.GetNNormalBins(), 3)
        self.assertEqual(axis.GetTotalNBins(), 3)
        self.assertFalse(axis.HasOverflowBin())

    def test_GetNormalRange(self):
        categories = ["a", "b", "c"]
        axis = RCategoricalAxis(categories)
        self.assertEqual(len(list(axis.GetNormalRange())), 3)
        self.assertEqual(len(list(axis.GetNormalRange(1, 2))), 1)

    def test_GetFullRange(self):
        categories = ["a", "b", "c"]
        axis = RCategoricalAxis(categories)
        self.assertEqual(len(list(axis.GetFullRange())), 4)


class AxisVariant(unittest.TestCase):
    def test_RegularAxis(self):
        Bins = 20
        axis = RAxisVariant(RRegularAxis(Bins, (0, Bins)))
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins + 2)

    def test_VariableBinAxis(self):
        Bins = 20
        bins = [i for i in range(0, Bins + 1)]
        axis = RAxisVariant(RVariableBinAxis(bins))
        self.assertEqual(axis.GetNNormalBins(), Bins)
        self.assertEqual(axis.GetTotalNBins(), Bins + 2)

    def test_CategoricalAxis(self):
        categories = ["a", "b", "c"]
        axis = RAxisVariant(RCategoricalAxis(categories))
        self.assertEqual(axis.GetNNormalBins(), 3)
        self.assertEqual(axis.GetTotalNBins(), 4)


class HistEngine(unittest.TestCase):
    def test_init(self):
        BinsX = 20
        regularAxis = RRegularAxis(BinsX, (0, BinsX))
        BinsY = 30
        bins = [i for i in range(0, BinsY + 1)]
        variableBinAxis = RVariableBinAxis(bins)
        categories = ["a", "b", "c"]
        categoricalAxis = RCategoricalAxis(categories)

        engine = RHistEngine["int"](regularAxis, variableBinAxis, categoricalAxis)
        self.assertEqual(engine.GetNDimensions(), 3)
        self.assertEqual(engine.GetTotalNBins(), (BinsX + 2) * (BinsY + 2) * (len(categories) + 1))

        # The user can also construct a list of axes.
        axes = [regularAxis, variableBinAxis, categoricalAxis]
        engine = RHistEngine["int"](axes)
        self.assertEqual(engine.GetNDimensions(), 3)
        self.assertEqual(engine.GetTotalNBins(), (BinsX + 2) * (BinsY + 2) * (len(categories) + 1))

        # Construct a one-dimensional histogram engine with a regular axis.
        engine = RHistEngine["int"](BinsX, (0, BinsX))
        self.assertEqual(engine.GetNDimensions(), 1)
        self.assertEqual(engine.GetAxes()[0].GetNNormalBins(), BinsX)

    def test_Add(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engineA = RHistEngine["int"](axis)
        engineB = RHistEngine["int"](axis)
        engineC = RHistEngine["int"](axis)

        engineA.Fill(-100)
        for i in range(0, Bins):
            engineA.Fill(i + 0.5)
            engineA.Fill(i + 0.5)
            engineB.Fill(i + 0.5)
        engineB.Fill(100)

        engineC.Add(engineA)
        engineC.Add(engineB)

        engineA.Add(engineB)

        self.assertEqual(engineA.GetBinContent(RBinIndex.Underflow()), 1)
        self.assertEqual(engineB.GetBinContent(RBinIndex.Underflow()), 0)
        self.assertEqual(engineC.GetBinContent(RBinIndex.Underflow()), 1)
        for index in axis.GetNormalRange():
            self.assertEqual(engineA.GetBinContent(index), 3)
            self.assertEqual(engineB.GetBinContent(index), 1)
            self.assertEqual(engineC.GetBinContent(index), 3)
        self.assertEqual(engineA.GetBinContent(RBinIndex.Overflow()), 1)
        self.assertEqual(engineB.GetBinContent(RBinIndex.Overflow()), 1)
        self.assertEqual(engineC.GetBinContent(RBinIndex.Overflow()), 1)

    def test_Clear(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["int"](axis)

        engine.Fill(-100)
        for i in range(0, Bins):
            engine.Fill(i + 0.5)
        engine.Fill(100)

        engine.Clear()

        for index in axis.GetFullRange():
            self.assertEqual(engine.GetBinContent(index), 0)

    def test_Clone(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engineA = RHistEngine["int"](axis)

        engineA.Fill(-100)
        for i in range(0, Bins):
            engineA.Fill(i + 0.5)
        engineA.Fill(100)

        engineB = engineA.Clone()
        self.assertEqual(engineA.GetNDimensions(), 1)
        self.assertEqual(engineA.GetTotalNBins(), Bins + 2)

        self.assertEqual(engineB.GetBinContent(RBinIndex.Underflow()), 1)
        for index in axis.GetNormalRange():
            self.assertEqual(engineB.GetBinContent(index), 1)
        self.assertEqual(engineB.GetBinContent(RBinIndex.Overflow()), 1)

        # Check that we can continue filling the clone.
        for i in range(0, Bins):
            engineB.Fill(i + 0.5)

        for index in axis.GetNormalRange():
            self.assertEqual(engineA.GetBinContent(index), 1)
            self.assertEqual(engineB.GetBinContent(index), 2)

    def test_Fill(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["int"](axis)

        engine.Fill(-100)
        for i in range(0, Bins):
            engine.Fill(i + 0.5)
        engine.Fill(100)

        for index in axis.GetFullRange():
            self.assertEqual(engine.GetBinContent(index), 1)

    def test_FillWeight(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["float"](axis)

        engine.Fill(-100, RWeight(0.25))
        for i in range(0, Bins):
            engine.Fill(i + 0.5, RWeight(0.1 + i * 0.03))
        engine.Fill(100, RWeight(0.75))

        self.assertAlmostEqual(engine.GetBinContent(RBinIndex.Underflow()), 0.25)
        for index in axis.GetNormalRange():
            self.assertAlmostEqual(engine.GetBinContent(index), 0.1 + index.GetIndex() * 0.03)
        self.assertAlmostEqual(engine.GetBinContent(RBinIndex.Overflow()), 0.75)

    def test_Scale(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        engine = RHistEngine["float"](axis)

        engine.Fill(-100, RWeight(0.25))
        for i in range(0, Bins):
            engine.Fill(i + 0.5, RWeight(0.1 + i * 0.03))
        engine.Fill(100, RWeight(0.75))

        Factor = 0.8
        engine.Scale(Factor)

        self.assertAlmostEqual(engine.GetBinContent(RBinIndex.Underflow()), Factor * 0.25)
        for index in axis.GetNormalRange():
            self.assertAlmostEqual(engine.GetBinContent(index), Factor * (0.1 + index.GetIndex() * 0.03))
        self.assertAlmostEqual(engine.GetBinContent(RBinIndex.Overflow()), Factor * 0.75)


class HistStats(unittest.TestCase):
    def test_init(self):
        stats = RHistStats(1)
        self.assertEqual(stats.GetNDimensions(), 1)

    def test_Fill(self):
        stats = RHistStats(3)
        self.assertEqual(stats.GetNEntries(), 0)
        self.assertTrue(math.isnan(stats.ComputeNEffectiveEntries()))
        self.assertTrue(math.isnan(stats.ComputeMean()))
        self.assertTrue(math.isnan(stats.ComputeStdDev()))

        Entries = 20
        for i in range(0, Entries):
            stats.Fill(i, 2 * i, i * i)

        self.assertEqual(stats.GetNEntries(), Entries)
        self.assertAlmostEqual(stats.GetSumW(), Entries)
        self.assertAlmostEqual(stats.GetSumW2(), Entries)
        self.assertAlmostEqual(stats.ComputeNEffectiveEntries(), Entries)

        self.assertAlmostEqual(stats.ComputeMean(), 9.5)
        self.assertAlmostEqual(stats.ComputeMean(1), 19)
        self.assertAlmostEqual(stats.ComputeMean(2), 123.5)
        self.assertAlmostEqual(stats.ComputeStdDev(), math.sqrt(33.25))
        self.assertAlmostEqual(stats.ComputeStdDev(1), math.sqrt(133))
        self.assertAlmostEqual(stats.ComputeStdDev(2), math.sqrt(12881.05))

    def test_FillWeighted(self):
        stats = RHistStats(3)
        self.assertEqual(stats.GetNEntries(), 0)
        self.assertTrue(math.isnan(stats.ComputeNEffectiveEntries()))
        self.assertTrue(math.isnan(stats.ComputeMean()))
        self.assertTrue(math.isnan(stats.ComputeStdDev()))

        Entries = 20
        for i in range(0, Entries):
            stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i))

        self.assertEqual(stats.GetNEntries(), Entries)
        self.assertAlmostEqual(stats.GetSumW(), 7.7)
        self.assertAlmostEqual(stats.GetSumW2(), 3.563)
        # Cross-checked with TH1
        self.assertAlmostEqual(stats.ComputeNEffectiveEntries(), 16.640471512770137)

        self.assertAlmostEqual(stats.ComputeMean(), 12.090909090909090)
        self.assertAlmostEqual(stats.ComputeMean(1), 24.181818181818180)
        self.assertAlmostEqual(stats.ComputeMean(2), 172.72727272727272)
        self.assertAlmostEqual(stats.ComputeStdDev(), 5.15142602, places=5)
        self.assertAlmostEqual(stats.ComputeStdDev(1), 10.3028520, places=5)
        self.assertAlmostEqual(stats.ComputeStdDev(2), 114.266905, places=5)

    def test_Scale(self):
        stats = RHistStats(3)
        self.assertEqual(stats.GetNEntries(), 0)

        Entries = 20
        for i in range(0, Entries):
            stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i))

        Factor = 0.8
        stats.Scale(Factor)

        self.assertEqual(stats.GetNEntries(), Entries)
        self.assertAlmostEqual(stats.GetSumW(), Factor * 7.7)
        self.assertAlmostEqual(stats.GetSumW2(), Factor * Factor * 3.563)


class Hist(unittest.TestCase):
    def test_init(self):
        Bins = 20
        regularAxis = RRegularAxis(Bins, (0, Bins))

        hist = RHist["int"](regularAxis, regularAxis)
        self.assertEqual(hist.GetNDimensions(), 2)
        self.assertEqual(hist.GetEngine().GetNDimensions(), 2)
        self.assertEqual(hist.GetStats().GetNDimensions(), 2)
        self.assertEqual(hist.GetTotalNBins(), (Bins + 2) * (Bins + 2))

        # The user can also construct a list of axes.
        axes = [regularAxis, regularAxis]
        hist = RHist["int"](axes)
        self.assertEqual(hist.GetNDimensions(), 2)
        self.assertEqual(hist.GetTotalNBins(), (Bins + 2) * (Bins + 2))

        # Construct a one-dimensional histogram with a regular axis.
        hist = RHist["int"](Bins, (0, Bins))
        self.assertEqual(hist.GetNDimensions(), 1)
        self.assertEqual(hist.GetAxes()[0].GetNNormalBins(), Bins)

    def test_Add(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        histA = RHist["int"](axis)
        histB = RHist["int"](axis)

        histA.Fill(8.5)
        histB.Fill(9.5)

        histA.Add(histB)

        self.assertEqual(histA.GetNEntries(), 2)
        self.assertEqual(histA.GetBinContent(RBinIndex(8)), 1)
        self.assertEqual(histA.GetBinContent(RBinIndex(9)), 1)

    def test_Clear(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        hist = RHist["int"](axis)

        hist.Fill(8.5)
        hist.Fill(9.5)

        hist.Clear()

        self.assertEqual(hist.GetNEntries(), 0)
        self.assertEqual(hist.GetBinContent(RBinIndex(8)), 0)
        self.assertEqual(hist.GetBinContent(RBinIndex(9)), 0)

    def test_Clone(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        histA = RHist["int"](axis)

        histA.Fill(8.5)

        histB = histA.Clone()
        self.assertEqual(histA.GetNDimensions(), 1)
        self.assertEqual(histA.GetTotalNBins(), Bins + 2)

        # Check that we can continue filling the clone.
        histB.Fill(9.5)

        self.assertEqual(histA.GetNEntries(), 1)
        self.assertEqual(histB.GetNEntries(), 2)
        self.assertEqual(histA.GetBinContent(9), 0)
        self.assertEqual(histB.GetBinContent(9), 1)

    def test_Fill(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        hist = RHist["int"](axis)

        hist.Fill(8.5)
        hist.Fill(9.5)

        self.assertEqual(hist.GetBinContent(RBinIndex(8)), 1)
        self.assertEqual(hist.GetBinContent(9), 1)

        self.assertEqual(hist.GetNEntries(), 2)
        self.assertAlmostEqual(hist.ComputeNEffectiveEntries(), 2)
        self.assertAlmostEqual(hist.ComputeMean(), 9)
        self.assertAlmostEqual(hist.ComputeStdDev(), 0.5)

    def test_FillWeight(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        hist = RHist["float"](axis)

        hist.Fill(8.5, RWeight(0.8))
        hist.Fill(9.5, RWeight(0.9))

        self.assertAlmostEqual(hist.GetBinContent(RBinIndex(8)), 0.8)
        self.assertAlmostEqual(hist.GetBinContent(9), 0.9)

        self.assertEqual(hist.GetNEntries(), 2)
        self.assertAlmostEqual(hist.GetStats().GetSumW(), 1.7)
        self.assertAlmostEqual(hist.GetStats().GetSumW2(), 1.45)
        # Cross-checked with TH1
        self.assertAlmostEqual(hist.ComputeNEffectiveEntries(), 1.9931034)
        self.assertAlmostEqual(hist.ComputeMean(), 9.0294118)
        self.assertAlmostEqual(hist.ComputeStdDev(), 0.49913420)

    def test_Scale(self):
        Bins = 20
        axis = RRegularAxis(Bins, (0, Bins))
        hist = RHist["float"](axis)

        hist.Fill(8.5, RWeight(0.8))
        hist.Fill(9.5, RWeight(0.9))

        Factor = 0.8
        hist.Scale(Factor)

        self.assertAlmostEqual(hist.GetBinContent(8), Factor * 0.8)
        self.assertAlmostEqual(hist.GetBinContent(9), Factor * 0.9)

        self.assertEqual(hist.GetNEntries(), 2)
        self.assertAlmostEqual(hist.GetStats().GetSumW(), Factor * 1.7)
        self.assertAlmostEqual(hist.GetStats().GetSumW2(), Factor * Factor * 1.45)
        # Cross-checked with TH1 - unchanged compared to FillWeight because the factor cancels out.
        self.assertAlmostEqual(hist.ComputeNEffectiveEntries(), 1.9931034)
        self.assertAlmostEqual(hist.ComputeMean(), 9.0294118)
        self.assertAlmostEqual(hist.ComputeStdDev(), 0.49913420)
