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
from ROOT import gROOT, TH1F, SetMemoryPolicy


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
      kMemoryStrict = ROOT.kMemoryStrict
      
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

   def test2ObjectDestructionCallback( self ):
      """Test ROOT notification on object destruction"""

    # create ROOT traced object
      a = TH1F( 'memtest_th1f', 'title', 100, -1., 1. )

    # locate it
      self.assertTrue( a is gROOT.FindObject( 'memtest_th1f' ) )

    # destroy it
      del a

    # should no longer be accessible
      self.assertTrue( not gROOT.FindObject( 'memtest_th1f' ) )

   def set_mem_policy(self, callable_obj, pol):
      # Set the memory policy of the callable object received
      callable_obj.__mempolicy__ = pol

   def test3ObjectCallHeuristics( self ):
      """Test memory mgmt heuristics for object calls"""
      
      MemTester = ROOT.MemTester
      kMemoryStrict = ROOT.kMemoryStrict
      kMemoryHeuristics = ROOT.kMemoryHeuristics

    # This unit test assumes that the global memory policy is set to
    # "heuristics" at the beginning, so let's make sure of that.
      old_memory_policy = SetMemoryPolicy( kMemoryHeuristics )

    # reference calls should not give up ownership
      a = MemTester()
      self.assertEqual( MemTester.counter, 1 )
      MemTester.CallRef( a );
      self.assertEqual( MemTester.counter, 1 )

      self.set_mem_policy(MemTester.CallRef, kMemoryStrict)
      MemTester.CallRef( a );
      self.assertEqual( MemTester.counter, 1 )

      self.set_mem_policy(MemTester.CallRef, kMemoryHeuristics)
      MemTester.CallRef( a );
      self.assertEqual( MemTester.counter, 1 )

      del a
      self.assertEqual( MemTester.counter, 0 )

      MemTester.CallConstRef( MemTester() )
      self.assertEqual( MemTester.counter, 0 )

    # give up ownership in case of non-const pointer call only, unless overridden
      MemTester.CallConstPtr( MemTester() )
      self.assertEqual( MemTester.counter, 0 )

      self.set_mem_policy(MemTester.CallConstPtr, kMemoryStrict)
      MemTester.CallConstPtr( MemTester() )
      self.assertEqual( MemTester.counter, 0 )

      self.set_mem_policy(MemTester.CallConstPtr, kMemoryHeuristics)
      MemTester.CallConstPtr( MemTester() )
      self.assertEqual( MemTester.counter, 0 )

      b1 = MemTester()
      self.assertEqual( MemTester.counter, 1 )
      MemTester.CallPtr( b1 );
      self.assertEqual( MemTester.counter, 1 )
      del b1
      counter = 1
      self.assertEqual( MemTester.counter, counter )

      b2 = MemTester()
      counter += 1
      self.assertEqual( MemTester.counter, counter )
      SetMemoryPolicy( kMemoryStrict )
      MemTester.CallPtr( b2 );
      self.assertEqual( MemTester.counter, counter )
      del b2
      counter -= 1
      self.assertEqual( MemTester.counter, counter )

      b3 = MemTester()
      counter += 1
      self.assertEqual( MemTester.counter, counter )
      SetMemoryPolicy( kMemoryHeuristics )
      MemTester.CallPtr( b3 );
      self.assertEqual( MemTester.counter, counter )
      del b3
      self.assertEqual( MemTester.counter, counter )

      b4 = MemTester()
      counter += 1
      self.assertEqual( MemTester.counter, counter )
      self.set_mem_policy(MemTester.CallPtr, kMemoryStrict)
      MemTester.CallPtr( b4 );
      self.assertEqual( MemTester.counter, counter )
      del b4
      counter -= 1
      self.assertEqual( MemTester.counter, counter )

      b5 = MemTester()
      counter += 1
      self.assertEqual( MemTester.counter, counter )
      SetMemoryPolicy( kMemoryStrict )
      self.set_mem_policy(MemTester.CallPtr, kMemoryHeuristics)
      MemTester.CallPtr( b5 );
      self.assertEqual( MemTester.counter, counter )
      del b5
      self.assertEqual( MemTester.counter, counter )

    # test explicit destruction
      SetMemoryPolicy( kMemoryHeuristics )
      MemTester().counter = 1      # silly way of setting it to 0
      self.set_mem_policy(MemTester.CallPtr, kMemoryHeuristics)
      self.assertEqual( MemTester.counter, 0 )
      c = MemTester()
      self.assertEqual( MemTester.counter, 1 )
      MemTester.CallPtr( c );
      self.assertEqual( MemTester.counter, 1 )
      klass = gROOT.GetClass( 'MemTester' )
      klass.Destructor( c )
      self.assertEqual( MemTester.counter, 0 )
      del c             # c not derived from TObject, no notification
      self.assertEqual( MemTester.counter, 0 )

      SetMemoryPolicy( old_memory_policy )

   def test4DestructionOfDerivedClass( self ):
      """Derived classes should call base dtor automatically"""

      MemTester = ROOT.MemTester
      
      class D1( MemTester ):
         def __init__( self ):
            MemTester.__init__( self )

      self.assertEqual( MemTester.counter, 0 )
      d = D1()
      self.assertEqual( MemTester.counter, 1 )
      del d
      self.assertEqual( MemTester.counter, 0 )

      class D2( MemTester ):
         def __init__( self ):
            super( D2, self ).__init__()

      self.assertEqual( MemTester.counter, 0 )
      d = D2()
      self.assertEqual( MemTester.counter, 1 )
      del d
      self.assertEqual( MemTester.counter, 0 )



## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
