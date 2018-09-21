import unittest
from array import array

import ROOT
from libcppyy import SetOwnership
import numpy as np


class TTreeTChainSetBranchAddress(unittest.TestCase):
    """
    Test for the pythonization of TTree/TChain::SetBranchAddress, which allows to pass proxy
    references as arguments from the Python side. Example:
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
    def get_file_tree_and_chain(self):
        f = ROOT.TFile(self.filename)
        t = f.Get(self.treename)
        # Prevent double deletion of the tree (Python and C++ TFile)
        SetOwnership(t, False)

        c = ROOT.TChain(self.treename)
        c.Add(self.filename)
        c.Add(self.filename)

        return f,t,c

    # Tests
    # Basic type, array and struct leaf list do not actually need the pythonization,
    # but testing anyway for the sake of completeness
    def test_basic_type_branch(self):
        f,t,c = self.get_file_tree_and_chain()

        for ds in t,c:
            n = array('f', [ 0. ])
            ds.SetBranchAddress('floatb', n)
            ds.GetEntry(0)

            self.assertEqual(n[0], self.more)

    def test_array_branch(self):
        f,t,c = self.get_file_tree_and_chain()

        for ds in t,c:
            a = array('d', self.arraysize*[ 0. ])
            ds.SetBranchAddress('arrayb', a)
            ds.GetEntry(0)

            for j in range(self.arraysize):
                self.assertEqual(a[j], j)

    def test_numpy_array_branch(self):
        f,t,c = self.get_file_tree_and_chain()

        for ds in t,c:
            a = np.array(self.arraysize*[ 0. ]) # dtype='float64'
            ds.SetBranchAddress('arrayb', a)
            ds.GetEntry(0)

            for j in range(self.arraysize):
                self.assertEqual(a[j], j)

    def test_struct_branch_leaflist(self):
        f,t,c = self.get_file_tree_and_chain()

        for ds in t,c:
            ms = ROOT.MyStruct()
            ds.SetBranchAddress('structleaflistb', ms)
            ds.GetEntry(0)

            self.assertEqual(ms.myint1, self.more)
            self.assertEqual(ms.myint2, 0)

    # Vector and struct do benefit from the pythonization
    def test_vector_branch(self):
        f,t,c = self.get_file_tree_and_chain()

        for ds in t,c:
            v = ROOT.std.vector('double')()
            ds.SetBranchAddress('vectorb', v)
            ds.GetEntry(0)

            for j in range(self.arraysize):
                self.assertEqual(v[j], j)

    def test_struct_branch(self):
        f,t,c = self.get_file_tree_and_chain()

        for ds in t,c:
            ms = ROOT.MyStruct()
            ds.SetBranchAddress('structb', ms)
            ds.GetEntry(0)

            self.assertEqual(ms.myint1, self.more)
            self.assertEqual(ms.myint2, 0)


if __name__ == '__main__':
    unittest.main()
