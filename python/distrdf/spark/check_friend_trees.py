import os
import subprocess

from array import array

import pytest

import ROOT
from DistRDF.Backends import Spark


def check_histograms(h_parent, h_friend):
    """Check equality of histograms in tests"""
    # Both trees have the same number of entries, i.e. 10000
    assert h_parent.GetEntries() == 10000
    assert h_friend.GetEntries() == 10000

    # Check the mean of the distribution for each tree
    assert h_parent.GetMean() == pytest.approx(10, 0.01)
    assert h_friend.GetMean() == pytest.approx(20, 0.01)

    # Check the standard deviation of the distribution for each tree
    assert h_parent.GetStdDev() == pytest.approx(1, 0.01)
    assert h_friend.GetStdDev() == pytest.approx(1, 0.01)


class TestSparkFriendTrees:
    """Integration tests to check the working of DistRDF with friend trees"""

    def create_parent_tree(self, filename):
        """Creates a .root file with the parent TTree"""
        f = ROOT.TFile(filename, "recreate")
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

    def create_friend_tree(self, filename):
        """Creates a .root file with the friend TTree"""
        ff = ROOT.TFile(filename, "recreate")
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

    def test_tchain_with_friend_tchain_histo(self, connection):
        """
        Tests that the computational graph can be issued both on the
        parent chain and the friend chain.
        """

        main_filename = "main_chain.root"
        friend_filename = "friend_chain.root"

        self.create_parent_tree(main_filename)
        self.create_friend_tree(friend_filename)

        # Main TChain
        mainchain = ROOT.TChain("T")
        mainchain.Add(main_filename)

        # Friend TChain
        friendchain = ROOT.TChain("TF")
        friendchain.Add(friend_filename)

        # Add friend chain to the main one
        mainchain.AddFriend(friendchain)

        # Create a DistRDF RDataFrame with the main and the friend chains
        df = Spark.RDataFrame(mainchain, sparkcontext=connection)

        # Create histograms
        h_parent = df.Histo1D(("main", "main", 10, 0, 20), "x")
        h_friend = df.Histo1D(("friend", "friend", 10, 10, 30), "TF.x")

        check_histograms(h_parent, h_friend)

        # Remove unnecessary .root files
        os.remove(main_filename)
        os.remove(friend_filename)

    def test_ttree_with_friend_ttree_histo(self, connection):
        """
        Tests that the computational graph can be issued both on the
        parent tree and the friend tree.
        """
        main_filename = "main_tree.root"
        friend_filename = "friend_tree.root"

        self.create_parent_tree(main_filename)
        self.create_friend_tree(friend_filename)

        # Main TTree
        mainfile = ROOT.TFile(main_filename)
        maintree = mainfile.Get("T")

        # Friend TTree
        friendfile = ROOT.TFile(friend_filename)
        friendtree = friendfile.Get("TF")

        # Add friend tree to the main one
        maintree.AddFriend(friendtree)

        # Create a DistRDF RDataFrame with the main and the friend trees
        df = Spark.RDataFrame(maintree, sparkcontext=connection)

        # Create histograms
        h_parent = df.Histo1D(("main", "main", 10, 0, 20), "x")
        h_friend = df.Histo1D(("friend", "friend", 10, 10, 30), "TF.x")

        check_histograms(h_parent, h_friend)

        # Remove unnecessary .root files
        mainfile.Close()
        friendfile.Close()
        os.remove(main_filename)
        os.remove(friend_filename)

    def test_friends_tchain_noname_add_fullpath_addfriend_alias(self, connection):
        """Test against the reproducer of issue https://github.com/root-project/root/issues/7584"""

        rn1 = "rn1.root"
        rn2 = "rn2.root"
        friendsfilename = "friendtrees_spark.root"

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

        chain.Add("friendtrees_spark.root/randomNumbers")
        chainFriend.Add("friendtrees_spark.root/randomNumbersBis")

        chain.AddFriend(chainFriend, "myfriend")

        df = Spark.RDataFrame(chain, sparkcontext=connection)

        h_parent = df.Histo1D(("main", "main", 10, 0, 20), "rnd")
        h_friend = df.Histo1D(("friend", "friend", 10, 10, 30), "myfriend.rnd")

        check_histograms(h_parent, h_friend)

        os.remove(rn1)
        os.remove(rn2)
        os.remove(friendsfilename)


if __name__ == "__main__":
    pytest.main(args=[__file__])
