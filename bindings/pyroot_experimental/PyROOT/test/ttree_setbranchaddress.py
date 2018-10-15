import unittest
from array import array

import ROOT
from libcppyy import SetOwnership
import numpy as np


class TTreeSetBranchAddress(unittest.TestCase):
    """
    Test for the pythonization of TTree::SetBranchAddress, which allows to pass proxy
    references as arguments from the Python side. Example:
    `v = ROOT.std.vector('int')()`
    `t.SetBranchAddress("my_vector_branch", v)`

    Since this pythonization is common to TTree and its subclasses, TChain and TNtuple
    are also tested here.
    """

    filename  = 'treesetbranchaddress.root'
    treename  = 'mytree'
    tuplename = 'mytuple'
    nentries  = 1
    arraysize = 10
    more      = 10

    # Setup
    @classmethod
    def setUpClass(cls):
        ROOT.gInterpreter.Declare('#include "TreeHelper.h"')
        ROOT.CreateTTree(cls.filename,
                         cls.treename,
                         cls.nentries,
                         cls.arraysize,
                         cls.more,
                         "RECREATE")
        ROOT.CreateTNtuple(cls.filename,
                           cls.tuplename,
                           cls.nentries,
                           cls.more,
                           "UPDATE")

    # Helper
    def get_file_objects(self):
        f = ROOT.TFile(self.filename)
        t = f.Get(self.treename)
        # Prevent double deletion of the tree (Python and C++ TFile)
        SetOwnership(t, False)

        c = ROOT.TChain(self.treename)
        c.Add(self.filename)
        c.Add(self.filename)

        nt = f.Get(self.tuplename)
        SetOwnership(nt, False)

        return f,t,c,nt

    # Tests
    # Basic type, array and struct leaf list do not actually need the pythonization,
    # but testing anyway for the sake of completeness
    def test_basic_type_branch(self):
        f,t,c,nt = self.get_file_objects()
        
        # TTree, TChain
        for ds in t,c:
            n = array('f', [ 0. ])
            ds.SetBranchAddress('floatb', n)
            ds.GetEntry(0)

            self.assertEqual(n[0], self.more)

        # TNtuple
        colnames = ['x','y','z']
        cols = [ array('f', [ 0. ]) for _ in colnames ]
        ncols = len(cols)
        for i in range(ncols):
            nt.SetBranchAddress(colnames[i], cols[i])
        nt.GetEntry(0)
        for i in range(ncols):
            self.assertEqual(cols[i][0], i*self.more)

    def test_array_branch(self):
        f,t,c,_ = self.get_file_objects()

        for ds in t,c:
            a = array('d', self.arraysize*[ 0. ])
            ds.SetBranchAddress('arrayb', a)
            ds.GetEntry(0)

            for j in range(self.arraysize):
                self.assertEqual(a[j], j)

    def test_numpy_array_branch(self):
        f,t,c,_ = self.get_file_objects()

        for ds in t,c:
            a = np.array(self.arraysize*[ 0. ]) # dtype='float64'
            ds.SetBranchAddress('arrayb', a)
            ds.GetEntry(0)

            for j in range(self.arraysize):
                self.assertEqual(a[j], j)

    def test_struct_branch_leaflist(self):
        f,t,c,_ = self.get_file_objects()

        for ds in t,c:
            ms = ROOT.MyStruct()
            ds.SetBranchAddress('structleaflistb', ms)
            ds.GetEntry(0)

            self.assertEqual(ms.myint1, self.more)
            self.assertEqual(ms.myint2, 0)

    # Vector and struct do benefit from the pythonization
    def test_vector_branch(self):
        f,t,c,_ = self.get_file_objects()

        for ds in t,c:
            v = ROOT.std.vector('double')()
            ds.SetBranchAddress('vectorb', v)
            ds.GetEntry(0)

            for j in range(self.arraysize):
                self.assertEqual(v[j], j)

    def test_struct_branch(self):
        f,t,c,_ = self.get_file_objects()

        for ds in t,c:
            ms = ROOT.MyStruct()
            ds.SetBranchAddress('structb', ms)
            ds.GetEntry(0)

            self.assertEqual(ms.myint1, self.more)
            self.assertEqual(ms.myint2, 0)


if __name__ == '__main__':
    unittest.main()
