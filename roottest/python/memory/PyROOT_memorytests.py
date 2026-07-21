# File: roottest/python/memory/PyROOT_memorytests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 02/15/05
# Last: 06/15/15

"""Memory tests for PyROOT package."""

import os, sys, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if not os.path.exists('MemTester.C'):
    os.chdir(os.path.dirname(__file__))

import ROOT
from ROOT import gROOT


__all__ = [
   'Memory1TestCase'
]

class MyTestCase( unittest.TestCase ):
   def shortDescription( self ):
      desc = str(self)
      doc_first_line = None

      if self._testMethodDoc:
         doc_first_line = self._testMethodDoc.split("\n")[0].strip()
      if doc_first_line:
         desc = doc_first_line
      return desc


### Memory management test cases =============================================
class Memory1TestCase( MyTestCase ):

   def test1ObjectCreationDestruction( self ):
      """Test object creation and destruction"""

      gROOT.LoadMacro( 'MemTester.C+' )
      MemTester = ROOT.MemTester

      self.assertEqual( MemTester.counter, 0 )

    # test creation
      a = MemTester()
      self.assertEqual( MemTester.counter, 1 )

      b = MemTester()
      self.assertEqual( MemTester.counter, 2 )

   # tickle the objects a bit
      a.Dummy()
      c = b.Dummy

    # test destruction
      del a
      self.assertEqual( MemTester.counter, 1 )

      del b, c
      self.assertEqual( MemTester.counter, 0 )

   # NOTE: the former test3ObjectCallHeuristics and
   # test4DestructionOfDerivedClass were removed: they are covered upstream in
   # the cppyy test suite by test_regression.py::test44_heuristic_mem_policy
   # and test_crossinheritance.py (test12a_counter_test,
   # test13_virtual_dtors_and_del) respectively.



## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
