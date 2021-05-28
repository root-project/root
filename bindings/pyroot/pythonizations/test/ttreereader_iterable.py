import unittest

import ROOT


class TTreeReaderIterable(unittest.TestCase):
    """
    Test for the pythonization that makes instances of TTreeReader
    iterable in Python.
    """

    # Tests
    def test_iterable(self):
        # Create test tree
        filename = "ttreereader_iterable.root"
        rdf = ROOT.RDataFrame(10).Define("x", "int(rdfentry_)").Snapshot("t", filename)

        f = ROOT.TFile(filename)
        t = f.Get("t")
        r = ROOT.TTreeReader(t)
        x = ROOT.TTreeReaderValue['int'](r, "x")

        # Check correspondance between entry number returned by the iterator,
        # entry number of the reader and value of x
        for entry in r:
            self.assertEqual(entry, r.GetCurrentEntry())
            self.assertEqual(entry, x.__deref__())


if __name__ == '__main__':
    unittest.main()
