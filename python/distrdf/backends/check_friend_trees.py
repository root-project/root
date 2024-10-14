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

    main_filename = "../data/ttree/distrdf_roottest_check_friend_trees_main.root"
    friend_filename = "../data/ttree/distrdf_roottest_check_friend_trees_friend.root"

    def test_tchain_with_friend_tchain_histo(self, payload):
        """
        Tests that the computational graph can be issued both on the
        parent chain and the friend chain.
        """

        # Main TChain
        mainchain = ROOT.TChain("T")
        mainchain.Add(self.main_filename)

        # Friend TChain
        friendchain = ROOT.TChain("TF")
        friendchain.Add(self.friend_filename)

        # Add friend chain to the main one
        mainchain.AddFriend(friendchain)

        # Create a DistRDF RDataFrame with the main and the friend chains
        connection, _ = payload
        df = ROOT.RDataFrame(mainchain, executor=connection)

        # Create histograms
        h_parent = df.Histo1D(("main", "main", 10, 0, 20), "x")
        h_friend = df.Histo1D(("friend", "friend", 10, 10, 30), "TF.x")

        check_histograms(h_parent, h_friend)

    def test_ttree_with_friend_ttree_histo(self, payload):
        """
        Tests that the computational graph can be issued both on the
        parent tree and the friend tree.
        """

        with ROOT.TFile(self.main_filename) as mainfile, ROOT.TFile(self.friend_filename) as friendfile:
            maintree = mainfile.Get("T")
            friendtree = friendfile.Get("TF")

            maintree.AddFriend(friendtree)

            # Create a DistRDF RDataFrame with the main and the friend trees
            connection, _ = payload
            df = ROOT.RDataFrame(maintree, executor=connection)

            # Create histograms
            h_parent = df.Histo1D(("main", "main", 10, 0, 20), "x")
            h_friend = df.Histo1D(("friend", "friend", 10, 10, 30), "TF.x")

            check_histograms(h_parent, h_friend)

    def test_friends_tchain_noname_add_fullpath_addfriend_alias(self, payload):
        """Test against the reproducer of issue https://github.com/root-project/root/issues/7584"""

        # Test the specific case of a parent chain and friend chain with no
        # names, that receive one tree each in the form "filename/treename". The
        # friend is then added to the parent with an alias.
        chain = ROOT.TChain()
        chainFriend = ROOT.TChain()

        chain.Add(
            "../data/ttree/distrdf_roottest_check_friend_trees_7584.root/randomNumbers")
        chainFriend.Add(
            "../data/ttree/distrdf_roottest_check_friend_trees_7584.root/randomNumbersBis")

        chain.AddFriend(chainFriend, "myfriend")

        connection, _ = payload
        df = ROOT.RDataFrame(chain, executor=connection)

        h_parent = df.Histo1D(("main", "main", 10, 0, 20), "rnd")
        h_friend = df.Histo1D(("friend", "friend", 10, 10, 30), "myfriend.rnd")

        check_histograms(h_parent, h_friend)


if __name__ == "__main__":
    pytest.main(args=[__file__])
