import unittest
from array import array

import ROOT
from libcppyy import SetOwnership
import numpy as np


class TTreeBranch(unittest.TestCase):
    """
    Test for the pythonization of TTree::Branch, which allows to pass proxy
    references as arguments from the Python side. Example:
    `v = ROOT.std.vector('int')()`
    `t.Branch('my_vector_branch', v)`
    """

    filename  = 'treebranch.root'
    treename  = 'mytree'
    arraysize = 10
    ival      = 7
    fval      = 7.

    # Setup
    @classmethod
    def setUpClass(cls):
        ROOT.gInterpreter.Declare("""
        struct MyStruct {
            int myint1;
            int myint2;
        };
        """)

    # Helpers
    def create_file_and_tree(self):
        f = ROOT.TFile(self.filename, 'RECREATE')
        t = ROOT.TTree(self.treename, self.treename)
        # Prevent double deletion of the tree (Python and C++ TFile)
        SetOwnership(t, False)

        return f,t

    def get_tree(self):
        f = ROOT.TFile(self.filename)
        t = f.Get(self.treename)
        SetOwnership(t, False)

        return f,t

    @staticmethod
    def fill_and_close(f, t):
        t.Fill()
        f.Write()
        f.Close()

    # Tests
    # Basic type and array do not actually need the pythonization,
    # but testing anyway for the sake of completeness
    def test01_write_basic_type_branch(self):
        f,t = self.create_file_and_tree()

        n = array('f', [ self.fval ])
        t.Branch('floatb', n, 'floatb/F')

        self.fill_and_close(f, t)

    def test02_read_basic_type_branch(self):
        f,t = self.get_tree()
        
        for entry in t:
            self.assertEqual(entry.floatb, self.fval)

    def test03_write_array_branch(self):
        f,t = self.create_file_and_tree()

        a = array('d', self.arraysize*[ self.fval ])
        t.Branch('arrayb', a, 'arrayb[' + str(self.arraysize) + ']/D')

        self.fill_and_close(f, t)

    def test04_read_array_branch(self):
        f,t = self.get_tree()

        for entry in t:
            for elem in entry.arrayb:
                self.assertEqual(elem, self.fval)

    def test05_write_numpy_array_branch(self):
        f,t = self.create_file_and_tree()

        a = np.array(self.arraysize*[ self.fval ]) # dtype='float64'
        t.Branch('arrayb', a, 'arrayb[' + str(self.arraysize) + ']/D')

        self.fill_and_close(f, t)

    def test06_read_numpy_array_branch(self):
        f,t = self.get_tree()

        for entry in t:
            for elem in entry.arrayb:
                self.assertEqual(elem, self.fval)

    # Struct and vector do benefit from the pythonization
    def test07_write_struct_single_branch(self):
        f,t = self.create_file_and_tree()

        ms = ROOT.MyStruct()
        ms.myint1, ms.myint2 = self.ival, 2*self.ival
        t.Branch('structb', ms)

        self.fill_and_close(f, t)

    def test08_read_struct_single_branch(self):
        f,t = self.get_tree()

        for entry in t:
            ms = entry.structb
            self.assertEqual(ms.myint1, self.ival)
            self.assertEqual(ms.myint2, 2*self.ival)

    def test09_write_vector_branch(self):
        f,t = self.create_file_and_tree()

        v = ROOT.std.vector('double')(self.arraysize*[ self.fval ])
        t.Branch('vectorb', v)

        self.fill_and_close(f, t)

    def test10_read_vector_branch(self):
        f,t = self.get_tree()

        for entry in t:
            for elem in entry.vectorb:
                self.assertEqual(elem, self.fval)


if __name__ == '__main__':
    unittest.main()
