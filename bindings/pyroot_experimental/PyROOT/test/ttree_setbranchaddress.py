import unittest
from array import array

import ROOT
from libcppyy import SetOwnership


class TTreeSetBranchAddress(unittest.TestCase):
    """
    Test for the pythonization of TTree::SetBranchAddress, which allows to pass proxy references
    as arguments from the Python side. Example:
    `v = ROOT.std.vector('int')()`
    `t.SetBranchAddress("my_vector_branch", v)`
    """

    filename  = 'treesetbranchaddress.root'
    treename  = 'mytree'
    nentries  = 1
    arraysize = 10
    more      = 10

    # Setup
    @classmethod
    def setUpClass(cls):
        ROOT.gInterpreter.Declare('#include "TreeHelper.h"')
        ROOT.CreateTTree(cls.filename, cls.treename, cls.nentries, cls.arraysize, cls.more)

    # Helper
    def get_file_and_tree(self):
        f = ROOT.TFile(self.filename)
        t = f.Get(self.treename)
        # Prevent double deletion of the tree (Python and C++ TFile)
        SetOwnership(t, False)

        return f,t

    # Tests
    # Basic type, array and struct leaf list do not actually need the pythonization,
    # but testing anyway for the sake of completeness
    def test_basic_type_branch(self):
        f,t = self.get_file_and_tree()

        n = array('f', [ 0. ])
        t.SetBranchAddress('floatb', n)
        t.GetEntry(0)

        self.assertEqual(n[0], self.more)

    def test_array_branch(self):
        f,t = self.get_file_and_tree()

        a = array('d', self.arraysize*[ 0. ])
        t.SetBranchAddress('arrayb', a)
        t.GetEntry(0)

        for j in range(self.arraysize):
            self.assertEqual(a[j], j)

    def test_struct_branch_leaflist(self):
        f,t = self.get_file_and_tree()

        ms = ROOT.MyStruct()
        t.SetBranchAddress('structleaflistb', ms)
        t.GetEntry(0)

        self.assertEqual(ms.myint1, self.more)
        self.assertEqual(ms.myint2, 0)

    # Vector and struct do benefit from the pythonization
    def test_vector_branch(self):
        f,t = self.get_file_and_tree()

        v = ROOT.std.vector('double')()
        t.SetBranchAddress('vectorb', v)
        t.GetEntry(0)

        for j in range(self.arraysize):
            self.assertEqual(v[j], j)

    def test_struct_branch(self):
        f,t = self.get_file_and_tree()

        ms = ROOT.MyStruct()
        t.SetBranchAddress('structb', ms)
        t.GetEntry(0)

        self.assertEqual(ms.myint1, self.more)
        self.assertEqual(ms.myint2, 0)


if __name__ == '__main__':
    unittest.main()
