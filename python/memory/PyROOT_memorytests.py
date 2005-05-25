# File: roottest/python/memory/PyROOT_memorytests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 02/15/05
# Last: 05/05/05

"""Memory tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Memory1TestCase'
]


### Memory management test cases =============================================
class Memory1TestCase( unittest.TestCase ):
   def test1ObjectCreationDestruction( self ):
      """Test object creation and destruction"""

      gROOT.LoadMacro( 'MemTester.C+' )
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

   def test2ObjectDestuctionCallback( self ):
      """Test ROOT notification on object destruction"""

    # create ROOT traced object
      a = TH1F( 'memtest_th1f', 'title', 100, -1., 1. )

    # locate it
      self.assertEqual( a, gROOT.FindObject( 'memtest_th1f' ) )

    # destroy it
      del a

    # should no longer be accessible
      self.assert_( not gROOT.FindObject( 'memtest_th1f' ) )

   def test3ObjectCallHeuristics( self ):
      """Test memory mgmt heuristics for object calls"""

    # reference calls should not give up ownership
      a = MemTester()
      self.assertEqual( MemTester.counter, 1 )
      MemTester.CallRef( a );
      self.assertEqual( MemTester.counter, 1 )
      del a
      self.assertEqual( MemTester.counter, 0 )

      MemTester.CallConstRef( MemTester() )
      self.assertEqual( MemTester.counter, 0 )

    # give up ownership in case of non-const pointer call only
      MemTester.CallConstPtr( MemTester() )
      self.assertEqual( MemTester.counter, 0 )

      b = MemTester()
      self.assertEqual( MemTester.counter, 1 )
      MemTester.CallPtr( b );
      self.assertEqual( MemTester.counter, 1 )
      del b
      self.assertEqual( MemTester.counter, 1 )

    # test explicit destruction
      MemTester().counter = 1      # silly way of setting it to 0
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

## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
