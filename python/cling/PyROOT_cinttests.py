# File: roottest/python/cint/PyROOT_cinttests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 05/11/05
# Last: 05/11/05

"""CINT compatability tests for PyROOT package."""

import os, sys, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import FIXCLING

__all__ = [
   'CINT1ErrorTranslationTestCase'
]


### CINT error translation test cases ========================================
class CINT1ErrorTranslationTestCase( unittest.TestCase ):
   def test1IndexError( self ):
      """Test CINT index error translation"""

      if FIXCLING:
         return

      self.assertRaises( IndexError, gROOT.ProcessLine, "char aap[5]; aap[6] = \'\\0\';" )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
