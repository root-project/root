import os
import unittest
from array import array

import ROOT
from DistRDF.HeadNode import Factory


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
        friend_tree_alias = "TF"
        friend_tree_filename = "treefriend.root"
        friendtree = ROOT.TChain(friend_tree_name)
        friendtree.Add(friend_tree_filename)

        # Add friendTree to the parent
        basetree.AddFriend(friendtree)

        # Instantiate head node of the graph with the base TTree
        headnode = Factory.get_headnode(basetree)

        # Retrieve information about friends
        friendnamesalias, friendfilenames, friendchainsubnames = headnode.get_friendinfo()

        # Check that FriendInfo has non-empty lists
        self.assertIsNotNone(friendnamesalias)
        self.assertIsNotNone(friendfilenames)
        self.assertIsNotNone(friendchainsubnames)

        # Check that the three lists with treenames, filenames and subnames are populated
        # as expected.
        self.assertTupleEqual(friendnamesalias, ((friend_tree_name, friend_tree_alias),))
        self.assertTupleEqual(friendfilenames,  ((friend_tree_filename,),))
        self.assertTupleEqual(friendchainsubnames,  ((friend_tree_name,),))

        # Remove unnecessary .root files
        os.remove(base_tree_filename)
        os.remove(friend_tree_filename)
