import ROOT
import unittest
import PyRDF
from array import array
import os


class SparkFriendTreesTest(unittest.TestCase):
    """Integration tests to check the working of PyRDF with friend trees"""
    def tearDown(self):
        """Clean up the `SparkContext` objects that were created."""
        from PyRDF import current_backend
        current_backend.sparkContext.stop()

    def create_parent_tree(self):
        """Creates a .root file with the parent TTree"""
        f = ROOT.TFile("treeparent.root", "recreate")
        T = ROOT.TTree("T", "test friend trees")

        x = array("f", [0])
        T.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The parent will have a gaussian distribution with mean 10 and
        # standard deviation 1
        for i in range(10000):
            x[0] = r.Gaus(10, 1)
            T.Fill()

        f.Write()
        f.Close()

    def create_friend_tree(self):
        """Creates a .root file with the friend TTree"""
        ff = ROOT.TFile("treefriend.root", "recreate")
        TF = ROOT.TTree("TF", "tree friend")

        x = array("f", [0])
        TF.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The friend will have a gaussian distribution with mean 20 and
        # standard deviation 1
        for i in range(10000):
            x[0] = r.Gaus(20, 1)
            TF.Fill()

        ff.Write()
        ff.Close()

    def test_friend_tree_histo(self):
        """
        Tests that the computational graph can be issued both on the
        parent tree and the friend tree.
        """
        self.create_parent_tree()
        self.create_friend_tree()

        # Parent Tree
        baseTree = ROOT.TChain("T")
        baseTree.Add("treeparent.root")

        # Friend Tree
        friendTree = ROOT.TChain("TF")
        friendTree.Add("treefriend.root")

        # Add friendTree to the parent
        baseTree.AddFriend(friendTree)

        # Create a PyRDF RDataFrame with the parent and the friend trees
        PyRDF.use("spark")
        df = PyRDF.RDataFrame(baseTree)

        # Create histograms
        h_parent = df.Histo1D("x")
        h_friend = df.Histo1D("TF.x")

        # Both trees have the same number of entries, i.e. 10000
        self.assertEqual(h_parent.GetEntries(), 10000)
        self.assertEqual(h_friend.GetEntries(), 10000)

        # Check the mean of the distribution for each tree
        self.assertAlmostEqual(h_parent.GetMean(), 10, delta=0.01)
        self.assertAlmostEqual(h_friend.GetMean(), 20, delta=0.01)

        # Check the standard deviation of the distribution for each tree
        self.assertAlmostEqual(h_parent.GetStdDev(), 1, delta=0.01)
        self.assertAlmostEqual(h_friend.GetStdDev(), 1, delta=0.01)

        # Remove unnecessary .root files
        os.remove("treeparent.root")
        os.remove("treefriend.root")


if __name__ == "__main__":
    unittest.main()
