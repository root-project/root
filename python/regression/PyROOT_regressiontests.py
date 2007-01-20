# File: roottest/python/regression/PyROOT_regressiontests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/02/07
# Last: 01/16/07

"""Regression tests, lacking a better place, for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Regression1TwiceImportStarTestCase',
   'Regression2PyExceptionTestcase'
]


### "from ROOT import *" done in import-*-ed module ==========================
from Amir import *

class Regression1TwiceImportStarTestCase( unittest.TestCase ):
   def test1FromROOTImportStarInModule( self ):
      """Test handling of twice 'from ROOT import*'"""

      x = TestTChain()        # TestTChain defined in Amir.py


### "from ROOT import *" done in import-*-ed module ==========================
class Regression2PyExceptionTestcase( unittest.TestCase ):
   def test1RaiseAndTrapPyException( self ):
      """Test handling of a thrown TPyException object"""

      gROOT.LoadMacro( "Scott.C+" )

    # test of not overloaded global function
      self.assertRaises( SyntaxError, ThrowPyException )
      try:
         ThrowPyException()
      except SyntaxError, e:
         self.assertEqual( str(e), "test error message" )

    # test of overloaded function
      self.assertRaises( SyntaxError, MyThrowingClass.ThrowPyException, 1 )
      try:
         MyThrowingClass.ThrowPyException( 1 )
      except SyntaxError, e:
         self.assertEqual( str(e), "overloaded int test error message" )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
