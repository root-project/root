import os
import unittest
from array import array
import platform

import ROOT
from DistRDF.HeadNode import get_headnode, EmptySourceHeadNode


def create_dummy_headnode(*args):
    """Create dummy head node instance needed in the test"""
    # Pass None as `npartitions`. The tests will modify this member
    # according to needs
    return get_headnode(None, None, *args)


def fill_main_tree_and_indexed_friend(mainfile, auxfile):
    idx = array("i", [0])
    x = array("i", [0])
    y = array("i", [0])

    with ROOT.TFile(mainfile, "RECREATE") as f1:
        main_tree = ROOT.TTree("mainTree", "mainTree")
        main_tree.Branch("idx", idx, "idx/I")
        main_tree.Branch("x", x, "x/I")

        idx[0] = 1
        x[0] = 1
        main_tree.Fill()
        idx[0] = 1
        x[0] = 2
        main_tree.Fill()
        idx[0] = 1
        x[0] = 3
        main_tree.Fill()
        idx[0] = 2
        x[0] = 4
        main_tree.Fill()
        idx[0] = 2
        x[0] = 5
        main_tree.Fill()
        f1.WriteObject(main_tree, "mainTree")

    with ROOT.TFile(auxfile, "RECREATE") as f2:
        aux_tree = ROOT.TTree("auxTree", "auxTree")
        aux_tree.Branch("idx", idx, "idx/I")
        aux_tree.Branch("y", y, "y/I")

        idx[0] = 2
        y[0] = 5
        aux_tree.Fill()
        idx[0] = 1
        y[0] = 7
        aux_tree.Fill()
        f2.WriteObject(aux_tree, "auxTree")


class DataFrameConstructorTests(unittest.TestCase):
    """Check various functionalities of the HeadNode class"""

    @classmethod
    def setUpClass(cls):
        """Create a dummy file to use for the RDataFrame constructor."""
        cls.test_treename = "treename"
        cls.test_filenames = [
            "distrdf_constructors_file0.root", "distrdf_constructors_file1.root"]

        for fname in cls.test_filenames:
            with ROOT.TFile(fname, "RECREATE") as f:
                t = ROOT.TTree(cls.test_treename, cls.test_treename)
                f.WriteObject(t, cls.test_treename)

    @classmethod
    def tearDownClass(cls):
        for fname in cls.test_filenames:
            os.remove(fname)

    def test_incorrect_args(self):
        """Constructor with incorrect arguments"""
        with self.assertRaises(TypeError):
            # Incorrect first argument in 2-argument case
            create_dummy_headnode(10, "file.root")

        with self.assertRaises(TypeError):
            # Incorrect third argument in 3-argument case
            create_dummy_headnode(self.test_treename,
                                  self.test_filenames[0], "column1")

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

        # See https://github.com/root-project/root/issues/7541 and
        # https://bugs.llvm.org/show_bug.cgi?id=49692 :
        # llvm JIT fails to catch exceptions on M1, so we disable their testing
        if platform.processor() != "arm" or platform.mac_ver()[0] == '':
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
        rdf_2_files = self.test_filenames

        # Convert RDF files list to ROOT CPP vector
        reqd_vec = ROOT.std.vector("string")()
        for elem in rdf_2_files:
            reqd_vec.push_back(elem)

        # RDataFrame constructor with 2nd argument as string
        hn_1 = create_dummy_headnode(
            self.test_treename, self.test_filenames[0])

        # RDataFrame constructor with 2nd argument as Python list
        hn_2 = create_dummy_headnode(self.test_treename, rdf_2_files)

        # RDataFrame constructor with 2nd argument as ROOT CPP Vector
        hn_3 = create_dummy_headnode(self.test_treename, reqd_vec)

        for hn in (hn_1, hn_2, hn_3):
            self.assertEqual(hn.maintreename, self.test_treename)

        self.assertListEqual(hn_1.inputfiles, self.test_filenames[:1])
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
        hn_1 = create_dummy_headnode(
            self.test_treename, self.test_filenames[0], rdf_branches)

        # RDataFrame constructor with 3rd argument as ROOT CPP Vector
        hn_2 = create_dummy_headnode(
            self.test_treename, self.test_filenames[0], reqd_vec)

        for hn in (hn_1, hn_2):
            self.assertEqual(hn.maintreename, self.test_treename)
            self.assertListEqual(hn.inputfiles, self.test_filenames[:1])

        self.assertListEqual(hn_1.defaultbranches, rdf_branches)
        self.assertIsInstance(hn_2.defaultbranches, type(reqd_vec))
        self.assertListEqual(list(hn_2.defaultbranches), list(reqd_vec))

    def test_three_args_with_multiple_files(self):
        """Constructor with TTree, list of input files and selected branches"""
        rdf_branches = ["branch1", "branch2"]
        rdf_files = self.test_filenames

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
        hn_1 = create_dummy_headnode(
            self.test_treename, rdf_files, rdf_branches)

        # RDataFrame constructor with 2nd argument as Python List
        # and 3rd argument as ROOT CPP Vector
        hn_2 = create_dummy_headnode(
            self.test_treename, rdf_files, reqd_branches_vec)

        # RDataFrame constructor with 2nd argument as ROOT CPP Vector
        # and 3rd argument as Python List
        hn_3 = create_dummy_headnode(
            self.test_treename, reqd_files_vec, rdf_branches)

        # RDataFrame constructor with 2nd and 3rd arguments as ROOT
        # CPP Vectors
        hn_4 = create_dummy_headnode(
            self.test_treename, reqd_files_vec, reqd_branches_vec)

        for hn in (hn_1, hn_2, hn_3, hn_4):
            self.assertEqual(hn.maintreename, self.test_treename)
            self.assertListEqual(hn.inputfiles, rdf_files)
            self.assertListEqual(list(hn.defaultbranches), rdf_branches)

        self.assertIsInstance(hn_1.defaultbranches, type(rdf_branches))
        self.assertIsInstance(hn_3.defaultbranches, type(rdf_branches))
        self.assertIsInstance(hn_2.defaultbranches, type(reqd_branches_vec))
        self.assertIsInstance(hn_4.defaultbranches, type(reqd_branches_vec))

    def test_tree_with_friends_and_treeindex(self):
        """TTreeIndex is not supported in distributed mode."""
        # See https://github.com/root-project/root/issues/7541 and
        # https://bugs.llvm.org/show_bug.cgi?id=49692 :
        # llvm JIT fails to catch exceptions on M1, so we disable their testing
        if platform.processor() != "arm" or platform.mac_ver()[0] == '':
            main_file = "distrdf_indexed_friend_main.root"
            aux_file = "distrdf_indexed_friend_aux.root"
            fill_main_tree_and_indexed_friend(main_file, aux_file)

            main_chain = ROOT.TChain("mainTree", "mainTree")
            main_chain.Add(main_file)
            aux_chain = ROOT.TChain("auxTree", "auxTree")
            aux_chain.Add(aux_file)

            aux_chain.BuildIndex("idx")
            main_chain.AddFriend(aux_chain)

            with self.assertRaises(ValueError) as context:
                create_dummy_headnode(main_chain)

            self.assertEqual(str(context.exception),
                             "Friend tree 'auxTree' has a TTreeIndex. This is not supported in distributed mode.")

            # Remove unnecessary .root files
            os.remove(main_file)
            os.remove(aux_file)


class NumEntriesTest(unittest.TestCase):
    """'get_num_entries' returns the number of entries in the input dataset"""

    @classmethod
    def setUpClass(cls):
        """Create a dummy file to use for the RDataFrame constructor."""
        cls.test_treename = "treename"
        cls.test_filename = "distrdf_numentries_file.root"
        cls.test_tree_entries = 42

        with ROOT.TFile(cls.test_filename, "RECREATE") as f:
            tree = ROOT.TTree(cls.test_treename, cls.test_treename)

            x = array("f", [0])
            tree.Branch("b0", x, "b1/F")

            for i in range(cls.test_tree_entries):
                x[0] = i  # Change the vector element to 1
                tree.Fill()  # Fill the tree with that element

            f.WriteObject(tree, cls.test_treename)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_filename)

    def test_num_entries_with_rdatasetspec(self):
        """Compute number of entries from an RDatasetSpec-based headnode."""

        spec = ROOT.RDF.Experimental.RDatasetSpec()
        spec.AddSample(("", self.test_treename, self.test_filename))

        hn = create_dummy_headnode(spec)

        self.assertListEqual(hn.inputfiles, [self.test_filename])
        self.assertListEqual(hn.subtreenames, [self.test_treename])

        with ROOT.TFile(hn.inputfiles[0]) as f:
            t = f.Get(hn.subtreenames[0])
            self.assertEqual(t.GetEntries(), self.test_tree_entries)
