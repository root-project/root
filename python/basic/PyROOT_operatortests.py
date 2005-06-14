# File: roottest/python/basic/PyROOT_operatortests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 06/04/05

"""C++ operators interface unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Cpp1OperatorsTestCase'
]

gROOT.LoadMacro( "Operators.C+" )


### C++ operators overloading test cases =====================================
class Cpp1OperatorsTestCase( unittest.TestCase ):
   def test1MathOperators( self ):
      """Test overloading of math operators"""

      self.failUnlessEqual( Number(20) + Number(10), Number(30) )
      self.failUnlessEqual( Number(20) - Number(10), Number(10) )
      self.failUnlessEqual( Number(20) / Number(10), Number(2) )
      self.failUnlessEqual( Number(20) * Number(10), Number(200) )
      self.failUnlessEqual( Number(5)  & Number(14), Number(4) )
      self.failUnlessEqual( Number(5)  | Number(14), Number(15) )
      self.failUnlessEqual( Number(5)  ^ Number(14), Number(11) )
      self.failUnlessEqual( Number(5)  << 2, Number(20) )
      self.failUnlessEqual( Number(20) >> 2, Number(5) )

   def test2UnaryMathOperators( self ):
      """Test overloading of unary math operators"""

      n  = Number(20)
      n += Number(10)
      n -= Number(10)
      n *= Number(10)
      n /= Number(2)
      self.failUnlessEqual(n ,Number(100) )

   def test3ComparisonOperators( self ):
      """Test overloading of comparison operators"""

      self.failUnlessEqual( Number(20) >  Number(10), 1 )
      self.failUnlessEqual( Number(20) <  Number(10), 0 )
      self.failUnlessEqual( Number(20) >= Number(20), 1 )
      self.failUnlessEqual( Number(20) <= Number(10), 0 )
      self.failUnlessEqual( Number(20) != Number(10), 1 )
      self.failUnlessEqual( Number(20) == Number(10), 0 )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
