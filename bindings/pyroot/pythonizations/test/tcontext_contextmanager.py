import os
import unittest

import ROOT

from ROOT import TDirectory


class TContextContextManager(unittest.TestCase):
    """
    Test of TContext used as context manager
    """

    def default_constructor(self):
        """
        Check status of gDirectory with default constructor.
        """
        filename = "TContextContextManager_test_default_constructor.root"
        self.assertEqual(ROOT.gDirectory, ROOT.gROOT)

        with TDirectory.TContext():
            # Create a file to change gDirectory
            testfile = ROOT.TFile(filename, "recreate")
            self.assertEqual(ROOT.gDirectory, testfile)
            testfile.Close()

        self.assertEqual(ROOT.gDirectory, ROOT.gROOT)
        os.remove(filename)

    def constructor_onearg(self):
        """
        Check status of gDirectory with constructor taking a new directory.
        """
        filenames = ["TContextContextManager_test_constructor_onearg_{}.root".format(i) for i in range(2)]

        file0 = ROOT.TFile(filenames[0], "recreate")
        file1 = ROOT.TFile(filenames[1], "recreate")
        self.assertEqual(ROOT.gDirectory, file1)

        with TDirectory.TContext(file0):
            self.assertEqual(ROOT.gDirectory, file0)

        self.assertEqual(ROOT.gDirectory, file1)
        file0.Close()
        file1.Close()
        for filename in filenames:
            os.remove(filename)

    def constructor_twoargs(self):
        """
        Check status of gDirectory with constructor taking the previous directory and a new one.
        """
        filenames = ["TContextContextManager_test_constructor_onearg_{}.root".format(i) for i in range(3)]

        file0 = ROOT.TFile(filenames[0], "recreate")
        file1 = ROOT.TFile(filenames[1], "recreate")
        file2 = ROOT.TFile(filenames[2], "recreate")
        self.assertEqual(ROOT.gDirectory, file2)

        with TDirectory.TContext(file0, file1):
            self.assertEqual(ROOT.gDirectory, file1)

        self.assertEqual(ROOT.gDirectory, file0)
        file0.Close()
        file1.Close()
        file2.Close()
        for filename in filenames:
            os.remove(filename)

    def test_all(self):
        """
        Run all tests of this class sequentially.
        The tests of this class rely on the current directory, which can be changed
        unpredictably if they are run concurrently.
        """
        self.default_constructor()
        self.constructor_onearg()
        self.constructor_twoargs()


if __name__ == '__main__':
    unittest.main()
