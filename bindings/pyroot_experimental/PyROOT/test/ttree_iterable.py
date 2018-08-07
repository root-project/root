import unittest
from array import array

import ROOT
from libcppyy import SetOwnership


class TTreeIterable(unittest.TestCase):
    """
    Test for the pythonization that makes TTree instances iterable in Python. 
    For example, this allows to do:
    `for event in mytree:`
         `...`
    """

    filename  = 'treeiterable.root'
    treename  = 'mytree'
    nentries  = 10
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
    def test_basic_type_branch(self):
        f,t = self.get_file_and_tree()

        n = array('f', [ 0. ])
        t.SetBranchAddress('floatb', n)

        i = 0
        for entry in t:
            self.assertEqual(n[0], i+self.more)
            i += 1

    def test_array_branch(self):
        f,t = self.get_file_and_tree()

        a = array('d', self.arraysize*[ 0. ])
        t.SetBranchAddress('arrayb', a)

        i = 0
        for entry in t:
            for j in range(self.arraysize):
                self.assertEqual(a[j], i+j)
            i += 1

    def test_struct_branch(self):
        f,t = self.get_file_and_tree()

        ms = ROOT.MyStruct()
        t.SetBranchAddress('structleaflistb', ms)

        i = 0
        for entry in t:
            self.assertEqual(ms.myint1, i+self.more)
            self.assertEqual(ms.myint2, i*self.more)
            i += 1


if __name__ == '__main__':
    unittest.main()
