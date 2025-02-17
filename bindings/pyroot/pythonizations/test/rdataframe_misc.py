import cppyy
import platform
import unittest
import numpy
import os

import ROOT

class DatasetContext:
    """A helper class to create the dataset for the tutorial below."""

    filenames = [
        "rdataframe_misc_1.root",
        "rdataframe_misc_2.root",
        "rdataframe_misc_3.root"
    ]
    treename = "dataset"
    nentries = 5

    def __init__(self):
        for filename in self.filenames:
            with ROOT.TFile(filename, "RECREATE") as f:
                t = ROOT.TTree(self.treename, self.treename)

                x = numpy.array([0], dtype=int)
                y = numpy.array([0], dtype=int)
                t.Branch("x", x, "x/I")
                t.Branch("y", y, "y/I")

                for i in range(1, self.nentries + 1):
                    x[0] = i
                    y[0] = 2 * i
                    t.Fill()

                f.Write()

    def __enter__(self):
        """Enable using the class as a context manager."""
        return self

    def __exit__(self, *_):
        """
        Enable using the class as a context manager. At the end of the context,
        remove the files created.
        """
        for filename in self.filenames:
            os.remove(filename)

class RDataFrameMisc(unittest.TestCase):
    """Miscellaneous RDataFrame tests"""

    def test_empty_filenames(self):
        """
        An empty list of filenames should be detected and the user should be informed
        """
        # See https://github.com/root-project/root/issues/7541 and
        # https://bugs.llvm.org/show_bug.cgi?id=49692 :
        # llvm JIT fails to catch exceptions on MacOS ARM, so we disable their testing
        # Also fails on Windows for the same reason
        if (
            (platform.processor() != "arm" or platform.mac_ver()[0] == '') and not
            platform.system() == "Windows"
        ):
            # With implicit conversions, cppyy also needs to try dispatching to the various
            # constructor overloads. The C++ exception will be thrown, but will be incapsulated
            # in a more generic TypeError telling the user that none of the overloads worked
            with self.assertRaisesRegex(TypeError, "RDataFrame: empty list of input files."):
                ROOT.RDataFrame("events", [])

            # When passing explicitly the vector of strings, type dispatching will not be necessary
            # and the real C++ exception will immediately surface
            with self.assertRaisesRegex(cppyy.gbl.std.invalid_argument, "RDataFrame: empty list of input files."):
                ROOT.RDataFrame("events", ROOT.std.vector[ROOT.std.string]())

            with self.assertRaisesRegex(TypeError, "RDataFrame: empty list of input files."):
                ROOT.RDataFrame("events", ())

    def _get_rdf(self, dataset):
        chain = ROOT.TChain(dataset.treename)
        for filename in dataset.filenames:
            chain.Add(filename)

        return ROOT.RDataFrame(chain)
    
    def _get_chain(self, dataset):
        chain = ROOT.TChain(dataset.treename)
        for filename in dataset.filenames:
            chain.Add(filename)
        return chain

    def _define_col(self, rdf):
        return rdf.Define("z", "42")

    def _filter_x(self, rdf):
        return rdf.Filter("x > 2")
    
    def _test_rdf_in_function(self, chain):

        rdf = ROOT.RDataFrame(chain)
        meanx = rdf.Mean("x")
        meany = rdf.Mean("y")
        self.assertLess(meanx.GetValue(), meany.GetValue())

    def test_ttree_ownership(self):
        """
        Regression tests for https://github.com/root-project/root/issues/17691
        """
        # Issues on windows with contention on file deletion
        if platform.system() == "Windows":
            return

        with DatasetContext() as dataset:
            rdf = self._get_rdf(dataset)

            npy_dict = rdf.AsNumpy()
            self.assertIsNone(numpy.testing.assert_array_equal(npy_dict["x"], numpy.array([1,2,3,4,5]*3)))
            self.assertIsNone(numpy.testing.assert_array_equal(npy_dict["y"], numpy.array([2,4,6,8,10]*3)))

            chain = self._get_chain(dataset)

            rdf = ROOT.RDataFrame(chain)

            self._test_rdf_in_function(chain)

            rdf = self._define_col(rdf)
            rdf = self._filter_x(rdf)

            self.assertEqual(rdf.Count().GetValue(), 9)




if __name__ == '__main__':
    unittest.main()
