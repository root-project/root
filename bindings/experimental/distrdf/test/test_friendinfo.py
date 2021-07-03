import os
import unittest
from array import array

import ROOT
from DistRDF.HeadNode import get_headnode


class FriendInfoTest(unittest.TestCase):
    """Unit test for the FriendInfo class"""

    def create_parent_tree(self):
        """Creates a .root file with the parent TTree"""
        f = ROOT.TFile("treeparent.root", "recreate")
        t = ROOT.TTree("T", "test friend trees")

        x = array("f", [0])
        t.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The parent will have a gaussian distribution with mean 10 and
        # standard deviation 1
        for _ in range(10000):
            x[0] = r.Gaus(10, 1)
            t.Fill()

        f.Write()
        f.Close()

    def create_friend_tree(self):
        """Creates a .root file with the friend TTree"""
        ff = ROOT.TFile("treefriend.root", "recreate")
        tf = ROOT.TTree("TF", "tree friend")

        x = array("f", [0])
        tf.Branch("x", x, "x/F")

        r = ROOT.TRandom()
        # The friend will have a gaussian distribution with mean 20 and
        # standard deviation 1
        for _ in range(10000):
            x[0] = r.Gaus(20, 1)
            tf.Fill()

        ff.Write()
        ff.Close()

    def test_friend_info_with_ttree(self):
        """
        Check that RFriendInfo correctly stores information about the friend
        trees
        """
        self.create_parent_tree()
        self.create_friend_tree()

        # Parent Tree
        base_tree_name = "T"
        base_tree_filename = "treeparent.root"
        basetree = ROOT.TChain(base_tree_name)
        basetree.Add(base_tree_filename)

        # Friend Tree
        friend_tree_name = "TF"
        friend_tree_filename = "treefriend.root"
        friendtree = ROOT.TChain(friend_tree_name)
        friendtree.Add(friend_tree_filename)

        # Add friendTree to the parent
        basetree.AddFriend(friendtree)

        # Instantiate head node of the graph with the base TTree
        # Passing None as `npartitions` since it is not required for the test
        headnode = get_headnode(None, basetree)

        # Retrieve RFriendInfo instance
        friend_info = headnode._get_friend_info()

        # Convert to Python collections
        friendnamesalias = [(str(pair.first), str(pair.second)) for pair in friend_info.fFriendNames]
        friendfilenames = [[str(filename) for filename in filenames] for filenames in friend_info.fFriendFileNames]
        friendchainsubnames = [[str(chainsubname) for chainsubname in chainsubnames] for chainsubnames in friend_info.fFriendChainSubNames]

        # Check that the two lists with treenames and filenames are populated
        # as expected.
        self.assertListEqual(friendnamesalias, [(friend_tree_name, friend_tree_name)])
        self.assertListEqual(friendfilenames, [[friend_tree_filename]])
        self.assertListEqual(friendchainsubnames, [[friend_tree_name]])

        # Remove unnecessary .root files
        os.remove(base_tree_filename)
        os.remove(friend_tree_filename)
