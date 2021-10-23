import os
import subprocess
import unittest
from array import array

import ROOT
from DistRDF.Backends import Dask

from dask.distributed import Client, LocalCluster


class DaskFriendTreesTest(unittest.TestCase):
    """Integration tests to check the working of DistRDF with friend trees"""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """
        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""
        cls.client.shutdown()

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

        # Create a DistRDF RDataFrame with the parent and the friend trees
        df = Dask.RDataFrame(baseTree, daskclient=self.client)

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

    def test_friends_tchain_noname_add_fullpath_addfriend_alias(self):
        """Test against the reproducer of issue https://github.com/root-project/root/issues/7584"""

        rn1 = "rn1.root"
        rn2 = "rn2.root"
        friendsfilename = "friendtrees_dask.root"

        df_1 = ROOT.RDataFrame(10000)
        df_2 = ROOT.RDataFrame(10000)

        df_1 = df_1.Define("rnd", "gRandom->Gaus(10)")
        df_2 = df_2.Define("rnd", "gRandom->Gaus(20)")

        df_1.Snapshot("randomNumbers", rn1)
        df_2.Snapshot("randomNumbersBis", rn2)

        # Put the two trees together in a common file
        subprocess.run("hadd -f {} {} {}".format(friendsfilename, rn1, rn2),
                    shell=True, check=True)

        # Test the specific case of a parent chain and friend chain with no
        # names, that receive one tree each in the form "filename/treename". The
        # friend is then added to the parent with an alias.
        chain = ROOT.TChain()
        chainFriend = ROOT.TChain()

        chain.Add("friendtrees_dask.root/randomNumbers")
        chainFriend.Add("friendtrees_dask.root/randomNumbersBis")

        chain.AddFriend(chainFriend, "myfriend")

        df = Dask.RDataFrame(chain, daskclient=self.client)

        h_parent = df.Histo1D("rnd")
        h_friend = df.Histo1D("myfriend.rnd")

        self.assertEqual(h_parent.GetEntries(), 10000)
        self.assertEqual(h_friend.GetEntries(), 10000)

        self.assertAlmostEqual(h_parent.GetMean(), 10, delta=0.01)
        self.assertAlmostEqual(h_friend.GetMean(), 20, delta=0.01)

        self.assertAlmostEqual(h_parent.GetStdDev(), 1, delta=0.01)
        self.assertAlmostEqual(h_friend.GetStdDev(), 1, delta=0.01)

        os.remove(rn1)
        os.remove(rn2)
        os.remove(friendsfilename)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
