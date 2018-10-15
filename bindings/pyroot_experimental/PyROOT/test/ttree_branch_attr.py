import unittest

import ROOT
from libcppyy import SetOwnership


class TTreeBranchAttr(unittest.TestCase):
    """
    Test for the pythonization that allows to access top-level tree branches/leaves as attributes
    (i.e. `mytree.mybranch`)

    Since this pythonization is common to TTree and its subclasses, TChain is
    also tested here.
    """

    filename  = 'treebranchattr.root'
    treename  = 'mytree'
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
                         'RECREATE')

    # Helper
    def get_file_objects(self):
        f = ROOT.TFile(self.filename)
        t = f.Get(self.treename)
        # Prevent double deletion of the tree (Python and C++ TFile)
        SetOwnership(t, False)

        c = ROOT.TChain(self.treename)
        c.Add(self.filename)
        c.Add(self.filename)

        # Read first entry
        for ds in t,c:
            ds.GetEntry(0)

        return f,t,c

    # Tests
    def test_basic_type_branch(self):
        f,t,c = self.get_file_objects()

        for ds in t,c:
            self.assertEqual(ds.floatb, self.more)

    def test_array_branch(self):
        f,t,c = self.get_file_objects()

        for ds in t,c:
            a = ds.arrayb
        
            for j in range(self.arraysize):
                self.assertEqual(a[j], j)

    def test_vector_branch(self):
        f,t,c = self.get_file_objects()

        for ds in t,c:
            v = ds.vectorb

            for j in range(self.arraysize):
                self.assertEqual(v[j], j)

    def test_struct_branch(self):
        f,t,c = self.get_file_objects()

        for ds in t,c:
            ms = ds.structb

            self.assertEqual(ms.myint1, self.more)
            self.assertEqual(ms.myint2, 0)

    def test_struct_branch_leaflist(self):
        f,t,c = self.get_file_objects()

        for ds in t,c:
            self.assertEqual(ds.myintll1, self.more)
            self.assertEqual(ds.myintll2, 0)

    def test_alias_branch(self):
        f,t,c = self.get_file_objects()

        for ds in t,c:
            ds.SetAlias('myalias', 'floatb')

            self.assertEqual(ds.myalias, ds.floatb)


if __name__ == '__main__':
    unittest.main()
