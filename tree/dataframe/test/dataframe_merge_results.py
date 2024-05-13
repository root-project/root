import pickle
import unittest

import ROOT

GetMergeableValue = ROOT.ROOT.Detail.RDF.GetMergeableValue
MergeValues = ROOT.ROOT.Detail.RDF.MergeValues


class RDataFrameMergeResults(unittest.TestCase):
    """Various tests for the RMergeableValue family of classes."""

    def test_wrong_merge(self):
        """Merging results of different operations raises `TypeError`"""
        df = ROOT.RDataFrame(100)

        min = df.Min("rdfentry_")
        max = df.Max("rdfentry_")

        mmin = GetMergeableValue(min)
        mmax = GetMergeableValue(max)

        self.assertRaises(TypeError, MergeValues, mmin, mmax)

    def test_pickle_mean(self):
        """Merge two averages, pickle and merge again."""

        df1 = ROOT.RDataFrame(10)
        df2 = ROOT.RDataFrame(20)

        mean1 = df1.Mean("rdfentry_")
        mean2 = df2.Mean("rdfentry_")

        mm1 = GetMergeableValue(mean1)
        mm2 = GetMergeableValue(mean2)

        deser1 = pickle.loads(pickle.dumps(mm1))
        deser2 = pickle.loads(pickle.dumps(mm2))

        MergeValues(deser1, deser2)

        mergedcounts = 30.0  # float to help in the division
        mergedmean = (4.5 * 10 + 9.5 * 20) / mergedcounts

        self.assertAlmostEqual(deser1.GetValue(), mergedmean)

        deser_deser1 = pickle.loads(pickle.dumps(deser1))
        deser_deser2 = pickle.loads(pickle.dumps(deser2))

        MergeValues(deser_deser1, deser_deser2)

        mergedcounts = 50.0  # float to help in the division
        mergedmean = (mergedmean * 30 + 9.5 * 20) / mergedcounts

        self.assertAlmostEqual(deser_deser1.GetValue(), mergedmean)

    def test_pickle_histo1d(self):
        """Merge two histograms, pickle and merge again."""

        df = ROOT.RDataFrame(100)

        h = df.Histo1D("rdfentry_")

        mh1 = GetMergeableValue(h)
        mh2 = GetMergeableValue(h)

        deser1 = pickle.loads(pickle.dumps(mh1))
        deser2 = pickle.loads(pickle.dumps(mh2))

        MergeValues(deser1, deser2)

        mergedh = deser1.GetValue()

        self.assertEqual(mergedh.GetEntries(), 200)
        self.assertAlmostEqual(mergedh.GetMean(), 49.5)

        deser_deser1 = pickle.loads(pickle.dumps(deser1))
        deser_deser2 = pickle.loads(pickle.dumps(deser2))

        MergeValues(deser_deser1, deser_deser2)

        mergedh = deser_deser1.GetValue()

        self.assertEqual(mergedh.GetEntries(), 300)
        self.assertAlmostEqual(mergedh.GetMean(), 49.5)

    def test_pickle_variedhisto(self):
        """Merge 3 varied histograms, pickle and evaluate values."""

        df = ROOT.RDataFrame(10)
        df1 = df.Define("x", "1").Vary("x", "ROOT::RVecI{-1, 2}", ["down", "up"])
        h = df1.Histo1D("x")
        hx = ROOT.RDF.Experimental.VariationsFor(h)

        mh1 = GetMergeableValue(hx)
        mh2 = GetMergeableValue(hx)

        deser1 = pickle.loads(pickle.dumps(mh1))
        deser2 = pickle.loads(pickle.dumps(mh2))

        MergeValues(deser1, deser2)

        keys = deser1.GetKeys()

        expectedmeans = [1, -1, 2]

        expectedkeys = ["nominal", "x:down", "x:up"]
        self.assertListEqual([str(key) for key in keys], expectedkeys)

        for key, mean in zip(expectedkeys, expectedmeans):
            histo = deser1.GetVariation(key)
            self.assertEqual(histo.GetEntries(), 20)
            self.assertAlmostEqual(histo.GetMean(), mean)


if __name__ == "__main__":
    unittest.main()
