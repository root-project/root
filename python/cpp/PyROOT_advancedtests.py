# File: roottest/python/cpp/PyROOT_advancedtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 04/20/07

"""C++ advanced language interface unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Cpp1VirtualInheritenceTestCase',
   'Cpp2TemplateLookupTestCase',
   'Cpp3PassByNonConstRefTestCase'
]

gROOT.LoadMacro( "AdvancedCpp.C+" )


### C++ virtual inheritence test cases =======================================
class Cpp1InheritenceTestCase( unittest.TestCase ):
   def test1DataMembers( self ):
      """Test data member access when using virtual inheritence"""

    #-----
      b = B()
      self.assertEqual( b.m_a,         1 )
      self.assertEqual( b.m_da,      1.1 )
      self.assertEqual( b.m_b,         2 )
      self.assertEqual( b.m_db,      2.2 )

      b.m_a = 11
      self.assertEqual( b.m_a,        11 )

      b.m_da = 11.11
      self.assertEqual( b.m_da,    11.11 )

      b.m_b = 22
      self.assertEqual( b.m_a,        11 )
      self.assertEqual( b.m_da,    11.11 )
      self.assertEqual( b.m_b,        22 )
      self.assertEqual( b.GetValue(), 22 )

      b.m_db = 22.22
      self.assertEqual( b.m_db,    22.22 )

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


### C++ template tests =======================================================
class Cpp2TemplateLookupTestCase( unittest.TestCase ):
   def test1SingleInstantiatedTemplate( self ):
      """Test data member access for a templated class"""

      t1 = T1( int )( 32 )
      self.assertEqual( t1.value(), 32 )
      self.assertEqual( t1.m_t1, 32 )

      t1.m_t1 = 41
      self.assertEqual( t1.value(), 41 )
      self.assertEqual( t1.m_t1, 41 )

   def test2TemplateInstantiatedTemplate( self ):
      """Test data member access for a templated class instantiated with a template"""

      t2 = T2( T1( int ) )()
      t2.m_t2.m_t1 = 32
      self.assertEqual( t2.m_t2.value(), 32 )
      self.assertEqual( t2.m_t2.m_t1, 32 )


### C++ by-non-const-ref arguments tests =====================================
class Cpp3PassByNonConstRefTestCase( unittest.TestCase ):
   def test1TestPlaceHolders( self ):
      """Test usage of Long/Double place holders"""

      l = Long( 42L )
      self.assertEqual( l, 42L )
      self.assertEqual( l/7L, 6L )
      self.assertEqual( l*1L, l )

      import math
      d = Double( math.pi )
      self.assertEqual( d, math.pi )
      self.assertEqual( d*math.pi, math.pi*math.pi )

   def test2PassBuiltinsByNonConstRef( self ):
      """Test parameter passing of builtins through non-const reference"""

      l = Long( 42L )
      SetLongThroughRef( l, 41 )
      self.assertEqual( l, 41 )

      d = Double( 3.14 )
      SetDoubleThroughRef( d, 3.1415 )
      self.assertEqual( d, 3.1415 )

      i = Long( 42L )
      SetIntThroughRef( i, 13 )
      self.assertEqual( i, 13 )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
