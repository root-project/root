import os
import unittest
from array import array

import ROOT
from DistRDF.Node import FriendInfo
from DistRDF.Node import HeadNode


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

    def test_empty_friend_info(self):
        """Check that FriendInfo is initialized with two empty lists"""

        friend_info = FriendInfo()

        friend_names = friend_info.friend_names
        friend_file_names = friend_info.friend_file_names

        # Check that both lists in FriendInfo are empty
        self.assertTrue(len(friend_names) == 0)
        self.assertTrue(len(friend_file_names) == 0)

        # Check functioning of __bool__ method
        self.assertFalse(friend_info)

    def test_friend_info_with_ttree(self):
        """
        Check that FriendInfo correctly stores information about the friend
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
        headnode = HeadNode(basetree)

        # Retrieve FriendInfo instance
        friend_info = headnode._get_friend_info()

        # Check that FriendInfo has non-empty lists
        self.assertTrue(friend_info)

        # Check that the two lists with treenames and filenames are populated
        # as expected.
        self.assertListEqual(friend_info.friend_names, [friend_tree_name])
        self.assertListEqual(
            friend_info.friend_file_names,
            [[friend_tree_filename]]
        )

        # Remove unnecessary .root files
        os.remove(base_tree_filename)
        os.remove(friend_tree_filename)
