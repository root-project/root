import os
import unittest
from array import array

import ROOT
from DistRDF.HeadNode import get_headnode

def create_dummy_headnode(*args):
    """Create dummy head node instance needed in the test"""
    # Pass None as `npartitions`. The tests will modify this member
    # according to needs
    return get_headnode(None, None, *args)

class FriendInfoTest(unittest.TestCase):
    """Unit test for the FriendInfo class"""

    def create_main_tree(self, treename, filename):
        """Creates a .root file with the parent TTree"""
        f = ROOT.TFile(filename, "recreate")
        t = ROOT.TTree(treename, "parent tree")

        x = array("i", [0])
        t.Branch("x", x, "x/I")

        for i in range(9):
            x[0] = i
            t.Fill()

        f.Write()
        f.Close()

    def create_friend_tree(self, treename, filename):
        """Creates a .root file with the friend TTree"""
        ff = ROOT.TFile(filename, "recreate")
        tf = ROOT.TTree(treename, "friend tree")

        y = array("i", [0])
        tf.Branch("y", y, "y/I")

        # The friend will have a gaussian distribution with mean 20 and
        # standard deviation 1
        for i in range(3):
            y[0] = i
            tf.Fill()

        ff.Write()
        ff.Close()

    def test_friendinfo_with_ttree(self):
        """
        Check that RFriendInfo correctly stores information about the friend
        trees
        """
        # Parent Tree
        main_tree_name = "T"
        main_tree_filename = "treeparent.root"
        self.create_main_tree(main_tree_name, main_tree_filename)
        maintree = ROOT.TChain(main_tree_name)
        maintree.Add(main_tree_filename)

        # Friend Tree
        friend_tree_name = "TF"
        friend_tree_alias = "TF"
        friend_tree_filename = "treefriend.root"
        self.create_friend_tree(friend_tree_name, friend_tree_filename)
        friendtree = ROOT.TChain(friend_tree_name)
        friendtree.Add(friend_tree_filename)

        # Add friendTree to the parent
        maintree.AddFriend(friendtree, friend_tree_alias)

        # Instantiate head node of the graph with the base TTree
        headnode = create_dummy_headnode(maintree)

        # Retrieve information about friends
        friendinfo = headnode.friendinfo

        # Convert to Python collections
        friendnamesalias = [(str(pair.first), str(pair.second)) for pair in friendinfo.fFriendNames]
        friendfilenames = [[str(filename) for filename in filenames] for filenames in friendinfo.fFriendFileNames]
        friendchainsubnames = [[str(chainsubname) for chainsubname in chainsubnames] for chainsubnames in friendinfo.fFriendChainSubNames]

        # Check that the three lists with treenames, filenames and subnames are populated
        # as expected.
        self.assertListEqual(friendnamesalias, [(friend_tree_name, friend_tree_name)])
        self.assertListEqual(friendfilenames, [[friend_tree_filename]])
        self.assertListEqual(friendchainsubnames, [[friend_tree_name]])

        # Remove unnecessary .root files
        os.remove(main_tree_filename)
        os.remove(friend_tree_filename)


    def test_friendinfo_chain_with_subnames(self):
        """
        Check that RFriendInfo correctly stores information about the friend
        trees
        """
        # Parent Tree
        main_tree_name = "treeparent"
        main_tree_filename = "treeparent.root"
        self.create_main_tree(main_tree_name, main_tree_filename)
        maintree = ROOT.TChain(main_tree_name)
        maintree.Add(main_tree_filename)

        # Friend chain
        friendchainname = "treefriend"
        friendchain = ROOT.TChain(friendchainname)
        actualfriendchainfilenames = []
        actualfriendchainsubnames = []
        for i in range(1,4):
            friend_name = "treefriend" + str(i)
            friend_filename = friend_name + ".root"
            actualfriendchainfilenames.append(friend_filename)
            actualfriendchainsubnames.append(friend_name)
            self.create_friend_tree(friend_name, friend_filename)
            friendchain.Add(friend_filename + "/" + friend_name)

        # Add friendTree to the parent
        maintree.AddFriend(friendchain)

        # Instantiate head node of the graph with the base TTree
        headnode = create_dummy_headnode(maintree)

        # Retrieve information about friends
        friendinfo = headnode.friendinfo

        # Convert to Python collections
        friendnamesalias = [(str(pair.first), str(pair.second)) for pair in friendinfo.fFriendNames]
        friendfilenames = [[str(filename) for filename in filenames] for filenames in friendinfo.fFriendFileNames]
        friendchainsubnames = [[str(chainsubname) for chainsubname in chainsubnames] for chainsubnames in friendinfo.fFriendChainSubNames]

        # Check that the three lists with treenames, filenames and subnames are populated
        # as expected.
        self.assertListEqual(friendnamesalias, [(friendchainname, friendchainname)])
        self.assertListEqual(friendfilenames, [actualfriendchainfilenames])
        self.assertListEqual(friendchainsubnames, [actualfriendchainsubnames])

        # Remove unnecessary .root files
        os.remove(main_tree_filename)
        for filename in actualfriendchainfilenames:
            os.remove(filename)