import unittest
import ROOT
from PyRDF import RDataFrame
from PyRDF import RDataFrameException


class RDataFrameConstructorTests(unittest.TestCase):
    """Check various constructors of PyRDF's RDataFrame"""

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
        RDF = RDataFrame(10)

        self.assertListEqual(RDF.args, [10])

    def test_two_args(self):
        """Constructor with list of input files"""
        rdf_2_files = ["file1.root", "file2.root"]

        # Convert RDF files list to ROOT CPP vector
        reqd_vec = ROOT.std.vector('string')()
        for elem in rdf_2_files:
            reqd_vec.push_back(elem)

        # RDataFrame constructor with 2nd argument as string
        RDF_1 = RDataFrame("treename", "file.root")

        # RDataFrame constructor with 2nd argument as Python list
        RDF_2 = RDataFrame("treename", rdf_2_files)

        # RDataFrame constructor with 2nd argument as ROOT CPP Vector
        RDF_3 = RDataFrame("treename", reqd_vec)

        self.assertArgs(RDF_1.args, ["treename", "file.root"])
        self.assertArgs(RDF_2.args, ["treename", reqd_vec])
        self.assertArgs(RDF_3.args, ["treename", reqd_vec])

    def test_three_args_with_single_file(self):
        """Constructor with TTree, one input file and selected branches"""
        rdf_branches = ["branch1", "branch2"]

        # Convert RDF branches list to ROOT CPP Vector
        reqd_vec = ROOT.std.vector('string')()
        for elem in rdf_branches:
            reqd_vec.push_back(elem)

        # RDataFrame constructor with 3rd argument as Python list
        RDF_1 = RDataFrame("treename", "file.root", rdf_branches)

        # RDataFrame constructor with 3rd argument as ROOT CPP Vector
        RDF_2 = RDataFrame("treename", "file.root", reqd_vec)

        self.assertArgs(RDF_1.args, ["treename", "file.root", reqd_vec])
        self.assertArgs(RDF_2.args, ["treename", "file.root", reqd_vec])

    def test_three_args_with_multiple_files(self):
        """Constructor with TTree, list of input files and selected branches"""
        rdf_branches = ["branch1", "branch2"]
        rdf_files = ["file1.root", "file2.root"]

        # Convert RDF files list to ROOT CPP Vector
        reqd_files_vec = ROOT.std.vector('string')()
        for elem in rdf_files:
            reqd_files_vec.push_back(elem)

        # Convert RDF files list to ROOT CPP Vector
        reqd_branches_vec = ROOT.std.vector('string')()
        for elem in rdf_branches:
            reqd_branches_vec.push_back(elem)

        # RDataFrame constructor with 2nd argument as Python List
        # and 3rd argument as Python List
        RDF_1 = RDataFrame("treename", rdf_files, rdf_branches)

        # RDataFrame constructor with 2nd argument as Python List
        # and 3rd argument as ROOT CPP Vector
        RDF_2 = RDataFrame("treename", rdf_files, reqd_branches_vec)

        # RDataFrame constructor with 2nd argument as ROOT CPP Vector
        # and 3rd argument as Python List
        RDF_3 = RDataFrame("treename", reqd_files_vec, rdf_branches)

        # RDataFrame constructor with 2nd and 3rd arguments as ROOT
        # CPP Vectors
        RDF_4 = RDataFrame("treename", reqd_files_vec, reqd_branches_vec)

        reqd_args_list = ["treename", reqd_files_vec, reqd_branches_vec]
        self.assertArgs(RDF_1.args, reqd_args_list)
        self.assertArgs(RDF_2.args, reqd_args_list)
        self.assertArgs(RDF_3.args, reqd_args_list)
        self.assertArgs(RDF_4.args, reqd_args_list)

    def test_incorrect_args(self):
        """Constructor with incorrect arguments"""
        with self.assertRaises(RDataFrameException):
            # Incorrect first argument in 2-argument case
            RDataFrame(10, "file.root")

        with self.assertRaises(RDataFrameException):
            # Incorrect third argument in 3-argument case
            RDataFrame("treename", "file.root", "column1")

        with self.assertRaises(RDataFrameException):
            # No argument case
            RDataFrame()


class NumEntriesTest(unittest.TestCase):
    """'get_num_entries' returns the number of entries in the input dataset"""

    def fill_tree(self, size):
        """Stores an RDataFrame object of a given size in 'data.root'."""
        tdf = ROOT.ROOT.RDataFrame(size)
        tdf.Define("b1", "(double) tdfentry_").Snapshot("tree", "data.root")

    def test_num_entries_single_arg_case(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of a single integer argument to RDataFrame.

        """
        rdf = RDataFrame(123)  # Create RDataFrame instance

        self.assertEqual(rdf.get_num_entries(), 123)

    def test_num_entries_two_args_case(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of two arguments to RDataFrame constructor.

        """
        self.fill_tree(1111)  # Store RDataFrame object of size 1111
        files_vec = ROOT.std.vector('string')()
        files_vec.push_back("data.root")

        # Create RDataFrame instances
        rdf = RDataFrame("tree", "data.root")
        rdf_1 = RDataFrame("tree", ["data.root"])
        rdf_2 = RDataFrame("tree", files_vec)

        self.assertEqual(rdf.get_num_entries(), 1111)
        self.assertEqual(rdf_1.get_num_entries(), 1111)
        self.assertEqual(rdf_2.get_num_entries(), 1111)

    def test_num_entries_three_args_case(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of two arguments to RDataFrame constructor.

        """
        self.fill_tree(1234)  # Store RDataFrame object of size 1234
        branches_vec_1 = ROOT.std.vector('string')()
        branches_vec_2 = ROOT.std.vector('string')()
        branches_vec_1.push_back("b1")
        branches_vec_2.push_back("b2")

        # Create RDataFrame instances
        rdf = RDataFrame("tree", "data.root", ["b1"])
        rdf_1 = RDataFrame("tree", "data.root", ["b2"])
        rdf_2 = RDataFrame("tree", "data.root", branches_vec_1)
        rdf_3 = RDataFrame("tree", "data.root", branches_vec_2)

        self.assertEqual(rdf.get_num_entries(), 1234)
        self.assertEqual(rdf_1.get_num_entries(), 1234)
        self.assertEqual(rdf_2.get_num_entries(), 1234)
        self.assertEqual(rdf_3.get_num_entries(), 1234)

    def test_num_entries_with_ttree_arg(self):
        """
        Ensure that the number of entries recorded are correct in the case
        of RDataFrame constructor with a TTree.

        """
        tree = ROOT.TTree("tree", "test")  # Create tree
        v = ROOT.std.vector("int")(4)  # Create a vector of 0s of size 4
        tree.Branch("vectorb", v)  # Create branch to hold the vector

        for i in range(4):
            v[i] = 1  # Change the vector element to 1
            tree.Fill()  # Fill the tree with that element

        rdf = RDataFrame(tree)

        self.assertEqual(rdf.get_num_entries(), 4)
