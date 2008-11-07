# File: roottest/python/cpp/PyROOT_advancedtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 11/07/08

"""C++ advanced language interface unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Cpp1VirtualInheritenceTestCase',
   'Cpp2TemplateLookupTestCase',
   'Cpp3PassByNonConstRefTestCase',
   'Cpp4HandlingAbstractClassesTestCase',
   'Cpp5AssignToRefArbitraryClassTestCase'
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
    # self.assertEqual( b.GetValue(), 22 )

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
    # self.assertEqual( c.GetValue(), 33 )

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
    # self.assertEqual( d.GetValue(), 44 )

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
      self.assertEqual( GetB( d ), 22 )
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

   def test3TemplateInstantiationWithVectorOfFloat( self ):
      """Test template instantiation with a std::vector< float >"""

      gROOT.LoadMacro( "Template.C+" )

    # the following will simply fail if there is a naming problem (e.g. std::,
    # allocator<int>, etc., etc.); note the parsing required ...
      b = MyTemplatedClass( std.vector( float ) )()

      for i in range(5):
         b.m_b.push_back( i )
         self.assertEqual( round( b.m_b[i], 5 ), float(i) )

   def test4TemplateMemberFunctions( self ):
      """Test template member functions lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...

      m = MyTemplatedMethodClass()

      self.assertEqual( m.GetSize( 'char' )(),   m.GetCharSize() )
      self.assertEqual( m.GetSize( int )(),      m.GetIntSize() )
      self.assertEqual( m.GetSize( long )(),     m.GetLongSize() )
      self.assertEqual( m.GetSize( float )(),    m.GetFloatSize() )
      self.assertEqual( m.GetSize( 'double' )(), m.GetDoubleSize() )

      self.assertEqual( m.GetSize( 'MyDoubleVector_t' )(), m.GetVectorOfDoubleSize() )

   def test5TemplateGlobalFunctions( self ):
      """Test template global function lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...

      f = MyTemplatedFunction

      self.assertEqual( f( 1 ), 1 )
      self.assertEqual( type( f( 2 ) ), type( 2 ) )
      self.assertEqual( f( 3. ), 3. )
      self.assertEqual( type( f( 4. ) ), type( 4. ) )


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


### C++ abstract classes should behave normally, but be non-instatiatable ====
class Cpp4HandlingAbstractClassesTestCase( unittest.TestCase ):
   def test1ClassHierarchy( self ):
      """Test abstract class in a hierarchy"""

      self.assert_( issubclass( MyConcreteClass, MyAbstractClass ) )

      c = MyConcreteClass()
      self.assert_( isinstance( c, MyConcreteClass ) )
      self.assert_( isinstance( c, MyAbstractClass ) )

   def test2Instantiation( self ):
      """Test non-instatiatability of abstract classes"""

      self.assertRaises( TypeError, MyAbstractClass )


### Return by reference should call assignment operator ======================
class Cpp5AssignToRefArbitraryClassTestCase( unittest.TestCase ):
   def test1AssignToReturnByRef( self ):
      """Test assignment to an instance returned by reference"""

      a = std.vector( RefTester )()
      a.push_back( RefTester( 42 ) )

      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 42 )

      a[0] = RefTester( 33 )
      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 33 )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
