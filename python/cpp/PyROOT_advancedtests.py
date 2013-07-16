# File: roottest/python/cpp/PyROOT_advancedtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 06/27/11

"""C++ advanced language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import *

__all__ = [
   'Cpp01VirtualInheritenceTestCase',
   'Cpp02TemplateLookupTestCase',
   'Cpp03PassByNonConstRefTestCase',
   'Cpp04HandlingAbstractClassesTestCase',
   'Cpp05AssignToRefArbitraryClassTestCase',
   'Cpp06MathConvertersTestCase',
   'Cpp07GloballyOverloadedComparatorTestCase',
   'Cpp08GlobalArraysTestCase',
   'Cpp09LongExpressionsTestCase',
   'Cpp10StandardExceptionsTestCase',
]

gROOT.LoadMacro( "AdvancedCpp.C+" )


### C++ virtual inheritence test cases =======================================
class Cpp01InheritenceTestCase( MyTestCase ):
   def test1DataMembers( self ):
      """Test data member access when using virtual inheritence"""

    #-----
      b = PR_B()
      self.assertEqual( b.m_a,         1 )
      self.assertEqual( b.m_da,      1.1 )
      self.assertEqual( b.m_b,         2 )
      self.assertEqual( b.m_db,      2.2 )

      b.m_a = 11
      self.assertEqual( b.m_a,        11 )
      self.assertEqual( b.m_b,         2 )

      b.m_da = 11.11
      self.assertEqual( b.m_da,    11.11 )
      self.assertEqual( b.m_db,      2.2 )

      b.m_b = 22
      self.assertEqual( b.m_a,        11 )
      self.assertEqual( b.m_da,    11.11 )
      self.assertEqual( b.m_b,        22 )
      self.assertEqual( b.GetValue(), 22 )

      b.m_db = 22.22
      self.assertEqual( b.m_db,    22.22 )

      del b

    #-----
      c = PR_C()
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
      d = PR_D()
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
      """Test reference passing when using virtual inheritance"""

    #-----
      b = PR_B()
      b.m_a, b.m_b = 11, 22
      self.assertEqual( GetA( b ), 11 )
      self.assertEqual( GetB( b ), 22 )
      del b

    #-----
      c = PR_C()
      c.m_a, c.m_b, c.m_c = 11, 22, 33
      self.assertEqual( GetA( c ), 11 )
      self.assertEqual( GetB( c ), 22 )
      self.assertEqual( GetC( c ), 33 )
      del c

    #-----
      d = PR_D()
      d.m_a, d.m_b, d.m_c, d.m_d = 11, 22, 33, 44
      self.assertEqual( GetA( d ), 11 )
      self.assertEqual( GetB( d ), 22 )
      self.assertEqual( GetC( d ), 33 )
      self.assertEqual( GetD( d ), 44 )
      del d


### C++ template tests =======================================================
class Cpp02TemplateLookupTestCase( MyTestCase ):
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
      self.assertEqual( m.GetSize( pylong )(),   m.GetLongSize() )
      self.assertEqual( m.GetSize( float )(),    m.GetFloatSize() )
      self.assertEqual( m.GetSize( 'double' )(), m.GetDoubleSize() )

      if not FIXCLING:
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
class Cpp03PassByNonConstRefTestCase( MyTestCase ):
   def test1TestPlaceHolders( self ):
      """Test usage of Long/Double place holders"""

      l = Long( pylong(42) )
      self.assertEqual( l, pylong(42) )
      self.assertEqual( l/7, pylong(6) )
      self.assertEqual( l*pylong(1), l )

      import math
      d = Double( math.pi )
      self.assertEqual( d, math.pi )
      self.assertEqual( d*math.pi, math.pi*math.pi )

   def test2PassBuiltinsByNonConstRef( self ):
      """Test parameter passing of builtins through non-const reference"""

      l = Long( pylong(42) )
      SetLongThroughRef( l, 41 )
      self.assertEqual( l, 41 )

      d = Double( 3.14 )
      SetDoubleThroughRef( d, 3.1415 )
      self.assertEqual( d, 3.1415 )

      i = Long( pylong(42) )
      SetIntThroughRef( i, 13 )
      self.assertEqual( i, 13 )

   def test3PassBuiltinsByNonConstRef( self ):
      """Test parameter passing of builtins through const reference"""

      self.assertEqual( PassLongThroughConstRef( 42 ), 42 )
      self.assertEqual( PassDoubleThroughConstRef( 3.1415 ), 3.1415 )
      self.assertEqual( PassIntThroughConstRef( 42 ), 42 )


### C++ abstract classes should behave normally, but be non-instatiatable ====
class Cpp04HandlingAbstractClassesTestCase( MyTestCase ):
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
class Cpp05AssignToRefArbitraryClassTestCase( MyTestCase ):
   def test1AssignToReturnByRef( self ):
      """Test assignment to an instance returned by reference"""

      a = std.vector( RefTester )()
      a.push_back( RefTester( 42 ) )

      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 42 )

      a[0] = RefTester( 33 )
      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 33 )


### Check availability of math conversions ===================================
class Cpp06MathConvertersTestCase( MyTestCase ):
   def test1MathConverters( self ):
      """Test operator int/long/double incl. typedef"""

      a = Convertible()
      a.m_i = 1234
      a.m_d = 4321.

      self.assertEqual( int(a),     1234 )
      self.assertEqual( int(a),    a.m_i )
      self.assertEqual( pylong(a), a.m_i )

      self.assertEqual( float(a), 4321. )
      self.assertEqual( float(a), a.m_d )


### Check global operator== overload =========================================
class Cpp07GloballyOverloadedComparatorTestCase( MyTestCase ):
   def test1Comparator( self ):
      """Check that the global operator!=/== is picked up"""

      if FIXCLING:
         return

      a, b = Comparable(), Comparable()

      self.assertEqual( a, b )
      self.assertEqual( b, a )
      self.assert_( a.__eq__( b ) )
      self.assert_( b.__eq__( a ) )
      self.assert_( a.__ne__( a ) )
      self.assert_( b.__ne__( b ) )
      self.assertEqual( a.__eq__( b ), True )
      self.assertEqual( b.__eq__( a ), True )
      self.assertEqual( a.__eq__( a ), False )
      self.assertEqual( b.__eq__( b ), False )


### Check access to global array variables ===================================
class Cpp08GlobalArraysTestCase( MyTestCase ):
   def test1DoubleArray( self ):
      """Verify access to array of doubles"""

      self.assertEqual( myGlobalDouble, 12. )
      self.assertRaises( IndexError, myGlobalArray.__getitem__, 500 )


### Verify temporary handling for long expressions ===========================
class Cpp09LongExpressionsTestCase( MyTestCase ):
   def test1LongExpressionWithTemporary( self ):
      """Test life time of temporary in long expression"""

      self.assertEqual( SomeClassWithData.SomeData.s_numData, 0 )
      r = SomeClassWithData()
      self.assertEqual( SomeClassWithData.SomeData.s_numData, 1 )

      if FIXCLING:
         return

    # in this, GimeData() returns a datamember of the temporary result
    # from GimeCopy(); normal ref-counting would let it go too early
      self.assertEqual( r.GimeCopy().GimeData().s_numData, 2 )

      del r
      self.assertEqual( SomeClassWithData.SomeData.s_numData, 0 )


### Test usability of standard exceptions ====================================
class Cpp10StandardExceptionsTestCase( MyTestCase ):
   def test1StandardExceptionsAccessFromPython( self ):
      """Access C++ standard exception objects from python"""

      e = std.runtime_error( "runtime pb!!" )
      self.assert_( e )
      self.assertEqual( e.what(), "runtime pb!!" )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
