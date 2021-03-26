import os
import unittest
from array import array

import ROOT
from DistRDF.Node import HeadNode
from DistRDF.Node import TreeInfo


class FriendInfoTest(unittest.TestCase):
    """Unit test to retrieve information from a TTree and insert it in a TreeInfo namedtuple."""

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
        Check that correct information about the friend trees is stored in TreeInfo
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
        alias = "myfriendalias"
        basetree.AddFriend(friendtree, alias)

        # Instantiate head node of the graph with the base TTree
        headnode = HeadNode(basetree)

        # Retrieve FriendInfo instance
        treeinfo = headnode.get_treeinfo()

        # Check that FriendInfo has non-empty lists
        self.assertIsNotNone(treeinfo.friendnamesalias)
        self.assertIsNotNone(treeinfo.friendfilenames)

        # Check that the two lists with treenames and filenames are populated
        # as expected.
        # friendnamesalias and friendfilenames are std::vector, convert to list
        # and then check data is equal to expected values.
        friendnamesalias = [(str(pair.first), str(pair.second)) for pair in treeinfo.friendnamesalias]
        friendfilenames = [[str(filename) for filename in filelist] for filelist in treeinfo.friendfilenames]
        self.assertListEqual(friendnamesalias, [(friend_tree_name, alias)])
        self.assertListEqual(friendfilenames, [[friend_tree_filename]])

        # Remove unnecessary .root files
        os.remove(base_tree_filename)
        os.remove(friend_tree_filename)
