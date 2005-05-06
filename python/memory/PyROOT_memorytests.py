# File: roottest/python/memory/PyROOT_memorytests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 02/15/05
# Last: 03/18/05

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


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
