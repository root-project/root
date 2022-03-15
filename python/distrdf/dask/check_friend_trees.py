import os
import subprocess

from array import array

import pytest

import ROOT
from DistRDF.Backends import Dask


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


class TestDaskFriendTrees:
    """Integration tests to check the working of DistRDF with friend trees"""

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

    def test_friend_tree_histo(self, connection):
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
        df = Dask.RDataFrame(baseTree, daskclient=connection)

        # Create histograms
        h_parent = df.Histo1D("x")
        h_friend = df.Histo1D("TF.x")

        check_histograms(h_parent, h_friend)

        # Remove unnecessary .root files
        os.remove("treeparent.root")
        os.remove("treefriend.root")

    def test_friends_tchain_noname_add_fullpath_addfriend_alias(self, connection):
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

        df = Dask.RDataFrame(chain, daskclient=connection)

        h_parent = df.Histo1D("rnd")
        h_friend = df.Histo1D("myfriend.rnd")

        check_histograms(h_parent, h_friend)

        os.remove(rn1)
        os.remove(rn2)
        os.remove(friendsfilename)


if __name__ == "__main__":
    pytest.main(args=[__file__])
