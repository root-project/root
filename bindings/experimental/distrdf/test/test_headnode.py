import os
import unittest
from array import array

import ROOT
from DistRDF.HeadNode import get_headnode, EmptySourceHeadNode


def create_dummy_headnode(*args):
    """Create dummy head node instance needed in the test"""
    # Pass None as `npartitions`. The tests will modify this member
    # according to needs
    return get_headnode(None, *args)


class DataFrameConstructorTests(unittest.TestCase):
    """Check various functionalities of the HeadNode class"""

    def test_incorrect_args(self):
        """Constructor with incorrect arguments"""
        with self.assertRaises(TypeError):
            # Incorrect first argument in 2-argument case
            create_dummy_headnode(10, "file.root")

        with self.assertRaises(TypeError):
            # Incorrect third argument in 3-argument case
            create_dummy_headnode("treename", "file.root", "column1")

        with self.assertRaises(TypeError):
            # No argument case
            create_dummy_headnode()

    def test_inmemory_tree(self):
        """Constructor with an in-memory-only tree is not supported"""
        tree = ROOT.TTree("tree", "Tree in memory")
        x = array("i", [0])
        tree.Branch("x", x, "x/I")
        for i in range(100):
            x[0] = i
            tree.Fill()

        with self.assertRaises(ROOT.std.runtime_error):
            # Trees with no associated files are not supported
            create_dummy_headnode(tree)

    def assertArgs(self, args_list1, args_list2):
        """
        Asserts the arguments from 2 given
        arguments lists. Specifically for the cases :
        * [str, list or vector or str]
        * [str, list or vector or str, list or vector]

        """
        for elem1, elem2 in zip(args_list1, args_list2):
            # Check if the types are equal
            self.assertIsInstance(elem1, type(elem2))
            # (this has to be done because, vector and
            # list are iterables, but not of same type)

            # Check the contents
            self.assertListEqual(list(elem1), list(elem2))

    def test_integer_arg(self):
        """Constructor with number of entries"""
        hn = create_dummy_headnode(10)

        self.assertEqual(hn.nentries, 10)
        self.assertIsInstance(hn, EmptySourceHeadNode)

    def test_two_args(self):
        """Constructor with list of input files"""
        rdf_2_files = ["file1.root", "file2.root"]

        # Convert RDF files list to ROOT CPP vector
        reqd_vec = ROOT.std.vector("string")()
        for elem in rdf_2_files:
            reqd_vec.push_back(elem)

        # RDataFrame constructor with 2nd argument as string
        hn_1 = create_dummy_headnode("treename", "file.root")

        # RDataFrame constructor with 2nd argument as Python list
        hn_2 = create_dummy_headnode("treename", rdf_2_files)

        # RDataFrame constructor with 2nd argument as ROOT CPP Vector
        hn_3 = create_dummy_headnode("treename", reqd_vec)

        for hn in (hn_1, hn_2, hn_3):
            self.assertEqual(hn.maintreename, "treename")

        self.assertListEqual(hn_1.inputfiles, ["file.root"])
        self.assertListEqual(hn_2.inputfiles, rdf_2_files)
        # hn_3 got file names as std::vector<std::string> but the TreeHeadNode
        # instance stores it as list[str]
        self.assertListEqual(hn_3.inputfiles, rdf_2_files)

    def test_three_args_with_single_file(self):
        """Constructor with TTree, one input file and selected branches"""
        rdf_branches = ["branch1", "branch2"]

        # Convert RDF branches list to ROOT CPP Vector
        reqd_vec = ROOT.std.vector("string")()
        for elem in rdf_branches:
            reqd_vec.push_back(elem)

        # RDataFrame constructor with 3rd argument as Python list
        hn_1 = create_dummy_headnode("treename", "file.root", rdf_branches)

        # RDataFrame constructor with 3rd argument as ROOT CPP Vector
        hn_2 = create_dummy_headnode("treename", "file.root", reqd_vec)

        for hn in (hn_1, hn_2):
            self.assertEqual(hn.maintreename, "treename")
            self.assertListEqual(hn.inputfiles, ["file.root"])

        self.assertListEqual(hn_1.defaultbranches, rdf_branches)
        self.assertIsInstance(hn_2.defaultbranches, type(reqd_vec))
        self.assertListEqual(list(hn_2.defaultbranches), list(reqd_vec))

    def test_three_args_with_multiple_files(self):
        """Constructor with TTree, list of input files and selected branches"""
        rdf_branches = ["branch1", "branch2"]
        rdf_files = ["file1.root", "file2.root"]

        # Convert RDF files list to ROOT CPP Vector
        reqd_files_vec = ROOT.std.vector("string")()
        for elem in rdf_files:
            reqd_files_vec.push_back(elem)

        # Convert RDF files list to ROOT CPP Vector
        reqd_branches_vec = ROOT.std.vector("string")()
        for elem in rdf_branches:
            reqd_branches_vec.push_back(elem)

        # RDataFrame constructor with 2nd argument as Python List
        # and 3rd argument as Python List
        hn_1 = create_dummy_headnode("treename", rdf_files, rdf_branches)

        # RDataFrame constructor with 2nd argument as Python List
        # and 3rd argument as ROOT CPP Vector
        hn_2 = create_dummy_headnode("treename", rdf_files, reqd_branches_vec)

        # RDataFrame constructor with 2nd argument as ROOT CPP Vector
        # and 3rd argument as Python List
        hn_3 = create_dummy_headnode("treename", reqd_files_vec, rdf_branches)

        # RDataFrame constructor with 2nd and 3rd arguments as ROOT
        # CPP Vectors
        hn_4 = create_dummy_headnode("treename", reqd_files_vec, reqd_branches_vec)

        for hn in (hn_1, hn_2, hn_3, hn_4):
            self.assertEqual(hn.maintreename, "treename")
            self.assertListEqual(hn.inputfiles, rdf_files)
            self.assertListEqual(list(hn.defaultbranches), rdf_branches)

        self.assertIsInstance(hn_1.defaultbranches, type(rdf_branches))
        self.assertIsInstance(hn_3.defaultbranches, type(rdf_branches))
        self.assertIsInstance(hn_2.defaultbranches, type(reqd_branches_vec))
        self.assertIsInstance(hn_4.defaultbranches, type(reqd_branches_vec))

class NumEntriesTest(unittest.TestCase):
    """'get_num_entries' returns the number of entries in the input dataset"""

    def fill_tree(self, size):
        """Stores an RDataFrame object of a given size in 'data.root'."""
        tdf = ROOT.ROOT.RDataFrame(size)
        tdf.Define("b1", "(double) tdfentry_").Snapshot("tree", "data.root")

    def test_num_entries_two_args_case(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of two arguments to RDataFrame constructor.

        """
        self.fill_tree(1111)  # Store RDataFrame object of size 1111
        files_vec = ROOT.std.vector("string")()
        files_vec.push_back("data.root")

        # Create RDataFrame instances
        hn = create_dummy_headnode("tree", "data.root")
        hn_1 = create_dummy_headnode("tree", ["data.root"])
        hn_2 = create_dummy_headnode("tree", files_vec)

        self.assertEqual(hn.tree.GetEntries(), 1111)
        self.assertEqual(hn_1.tree.GetEntries(), 1111)
        self.assertEqual(hn_2.tree.GetEntries(), 1111)

    def test_num_entries_three_args_case(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of two arguments to RDataFrame constructor.

        """
        self.fill_tree(1234)  # Store RDataFrame object of size 1234
        branches_vec_1 = ROOT.std.vector("string")()
        branches_vec_2 = ROOT.std.vector("string")()
        branches_vec_1.push_back("b1")
        branches_vec_2.push_back("b2")

        # Create RDataFrame instances
        hn = create_dummy_headnode("tree", "data.root", ["b1"])
        hn_1 = create_dummy_headnode("tree", "data.root", ["b2"])
        hn_2 = create_dummy_headnode("tree", "data.root", branches_vec_1)
        hn_3 = create_dummy_headnode("tree", "data.root", branches_vec_2)

        self.assertEqual(hn.tree.GetEntries(), 1234)
        self.assertEqual(hn_1.tree.GetEntries(), 1234)
        self.assertEqual(hn_2.tree.GetEntries(), 1234)
        self.assertEqual(hn_3.tree.GetEntries(), 1234)

    def test_num_entries_with_ttree_arg(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of RDataFrame constructor with a TTree.

        """
        filename = "test_num_entries_with_ttree_arg.root"
        f = ROOT.TFile(filename, "recreate")

        tree = ROOT.TTree("tree", "test")  # Create tree
        v = ROOT.std.vector("int")(4)  # Create a vector of 0s of size 4
        tree.Branch("vectorb", v)  # Create branch to hold the vector

        for i in range(4):
            v[i] = 1  # Change the vector element to 1
            tree.Fill()  # Fill the tree with that element

        f.Write()

        hn = create_dummy_headnode(tree)

        self.assertEqual(hn.tree.GetEntries(), 4)

        f.Close()
        os.remove(filename)
