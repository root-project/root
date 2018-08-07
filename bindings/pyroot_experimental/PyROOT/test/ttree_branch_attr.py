import unittest

import ROOT
from libcppyy import SetOwnership


class TTreeBranchAttr(unittest.TestCase):
    """
    Test for the pythonization that allows to access top-level tree branches/leaves as attributes
    (i.e. `mytree.mybranch`)
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
        ROOT.CreateTTree(cls.filename, cls.treename, cls.nentries, cls.arraysize, cls.more)

    # Helper
    def get_file_and_tree(self):
        f = ROOT.TFile(self.filename)
        t = f.Get(self.treename)
        # Prevent double deletion of the tree (Python and C++ TFile)
        SetOwnership(t, False)

        # Read first entry
        t.GetEntry(0)

        return f,t

    # Tests
    def test_basic_type_branch(self):
        f,t = self.get_file_and_tree()

        self.assertEqual(t.floatb, self.more)

    def test_array_branch(self):
        f,t = self.get_file_and_tree()

        a = t.arrayb
        
        for j in range(self.arraysize):
            self.assertEqual(a[j], j)

    def test_vector_branch(self):
        f,t = self.get_file_and_tree()

        v = t.vectorb

        for j in range(self.arraysize):
            self.assertEqual(v[j], j)

    def test_struct_branch(self):
        f,t = self.get_file_and_tree()

        ms = t.structb

        self.assertEqual(ms.myint1, self.more)
        self.assertEqual(ms.myint2, 0)

    def test_struct_branch_leaflist(self):
        f,t = self.get_file_and_tree()

        self.assertEqual(t.myintll1, self.more)
        self.assertEqual(t.myintll2, 0)

    def test_alias_branch(self):
        f,t = self.get_file_and_tree()

        t.SetAlias('myalias', 'floatb')

        self.assertEqual(t.myalias, t.floatb)


if __name__ == '__main__':
    unittest.main()
