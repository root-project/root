# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 06/04/05

"""C++ advanced language interface unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Cpp1VirtualInheritenceTestCase'
]

gROOT.LoadMacro( "AdvancedCpp.C+" )


### C++ virtual inheritence test cases =======================================
class Cpp1InheritenceTestCase( unittest.TestCase ):
   def test1DataMembers( self ):
      """Test data member access when using virtual inheritence"""

    #-----
      b = B()
      self.assertEqual( b.m_a,         1 )
      self.assertEqual( b.m_b,         2 )

      b.m_a = 11
      self.assertEqual( b.m_a,        11 )

      b.m_b = 22
      self.assertEqual( b.m_a,        11 )
      self.assertEqual( b.m_b,        22 )
      self.assertEqual( b.GetValue(), 22 )

      del b

    #-----
      c = C()
      self.assertEqual( c.m_a,         1 )
      self.assertEqual( c.m_b,         2 )
      self.assertEqual( c.m_c,         3 )

      c.m_a = 11
      self.assertEqual( c.m_a,        11 )

      c.m_b = 22
      self.assertEqual( c.m_a,        11 )
      self.assertEqual( c.m_b,        22 )

      c.m_c = 33
      self.assertEqual( c.m_a,        11 )
      self.assertEqual( c.m_b,        22 )
      self.assertEqual( c.m_c,        33 )
      self.assertEqual( c.GetValue(), 33 )

      del c

    #-----
      d = D()
      self.assertEqual( d.m_a,         1 )
      self.assertEqual( d.m_b,         2 )
      self.assertEqual( d.m_c,         3 )
      self.assertEqual( d.m_d,         4 )

      d.m_a = 11
      self.assertEqual( d.m_a,        11 )

      d.m_b = 22
      self.assertEqual( d.m_a,        11 )
      self.assertEqual( d.m_b,        22 )

      d.m_c = 33
      self.assertEqual( d.m_a,        11 )
      self.assertEqual( d.m_b,        22 )
      self.assertEqual( d.m_c,        33 )

      d.m_d = 44
      self.assertEqual( d.m_a,        11 )
      self.assertEqual( d.m_b,        22 )
      self.assertEqual( d.m_c,        33 )
      self.assertEqual( d.m_d,        44 )
      self.assertEqual( d.GetValue(), 44 )

      del d

   def test2PassByReference( self ):
      """Test reference passing when using virtual inheritence"""

    #-----
      b = B()
      b.m_a, b.m_b = 11, 22
      self.assertEqual( GetA( b ), 11 )
      self.assertEqual( GetB( b ), 22 )
      del b

    #-----
      c = C()
      c.m_a, c.m_b, c.m_c = 11, 22, 33
      self.assertEqual( GetA( c ), 11 )
      self.assertEqual( GetB( c ), 22 )
      self.assertEqual( GetC( c ), 33 )
      del c

    #-----
      d = D()
      d.m_a, d.m_b, d.m_c, d.m_d = 11, 22, 33, 44
      self.assertEqual( GetA( d ), 11 )
      #self.assertEqual( GetB( d ), 22 )
      self.assertEqual( GetC( d ), 33 )
      self.assertEqual( GetD( d ), 44 )
      del d


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
