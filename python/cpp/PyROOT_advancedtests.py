# File: roottest/python/cpp/PyROOT_advancedtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 05/04/15

"""C++ advanced language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import *

__all__ = [
   'Cpp01VirtualInheritence',
   'Cpp02TemplateLookup',
   'Cpp03PassByNonConstRef',
   'Cpp04HandlingAbstractClasses',
   'Cpp05AssignToRefArbitraryClass',
   'Cpp06MathConverters',
   'Cpp07GloballyOverloadedComparator',
   'Cpp08GlobalVariables',
   'Cpp09LongExpressions',
   'Cpp10StandardExceptions',
   'Cpp11PointerContainers',
   'Cpp12NamespaceLazyFunctions',
   'Cpp13OverloadedNewDelete',
   'Cpp14CopyConstructorOrdering',
]

gROOT.LoadMacro( "AdvancedCpp.C+" )


### C++ virtual inheritence test cases =======================================
class Cpp01Inheritence( MyTestCase ):
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
class Cpp02TemplateLookup( MyTestCase ):
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

    # template without arguments can not resolve; check that there is an exception
    # and a descriptive error message
      self.assertRaises( TypeError, m.GetSize )
      try:
         m.GetSize()
      except TypeError, e:
         self.assert_( "must be explicit" in str(e) )

      self.assertEqual( m.GetSize( 'char' )(),   m.GetCharSize() )
      self.assertEqual( m.GetSize( int )(),      m.GetIntSize() )
      self.assertEqual( m.GetSize( pylong )(),   m.GetLongSize() )
      self.assertEqual( m.GetSize( float )(),    m.GetFloatSize() )
      self.assertEqual( m.GetSize( 'double' )(), m.GetDoubleSize() )

      self.assertEqual( m.GetSize( 'MyDoubleVector_t' )(), m.GetVectorOfDoubleSize() )
      self.assertEqual( m.GetSize( 'vector<double>' )(), m.GetVectorOfDoubleSize() )

   def test5TemplateMemberFunctions( self ):
      """Test template member functions lookup and calls (set 2)"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...

      m = MyTemplatedMethodClass()

    # note that the function and template arguments are reverted
      self.assertRaises( TypeError, m.GetSize2( 'char', 'long' ), 'a', 1 )
      self.assertEqual( m.GetSize2( 'char', 'long' )( 1, 'a' ), m.GetCharSize() - m.GetLongSize() )
      self.assertEqual( m.GetSize2( 256L, 1. ), m.GetFloatSize() - m.GetLongSize() )

   def test6OverloadedTemplateMemberFunctions( self ):
      """Test overloaded template member functions lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...

      m = MyTemplatedMethodClass()

    # the number of entries in the class dir() is used to check whether
    # member templates have been instantiated on it
      nd = len(dir(MyTemplatedMethodClass))

    # use existing (note '-' to make sure the correct call was made
      self.assertEqual( m.GetSizeOL( 1 ),             -m.GetLongSize() )
      self.assertEqual( m.GetSizeOL( "aapje" ),       -len("aapje") )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd )

    # use existing explicit instantiations
      self.assertEqual( m.GetSizeOL( float )( 3.14 ),  m.GetFloatSize() )
      self.assertEqual( m.GetSizeOL( 3.14 ),           m.GetFloatSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd )

    # explicit forced instantiation
      self.assertEqual( m.GetSizeOL( int )( 1 ),       m.GetIntSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + 1 )
      self.assert_( 'GetSizeOL<int>' in dir(MyTemplatedMethodClass) )
      gzoi_id = id( MyTemplatedMethodClass.__dict__[ 'GetSizeOL<int>' ] )

    # second call should make no changes, but re-use
      self.assertEqual( m.GetSizeOL( int )( 1 ),       m.GetIntSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + 1 )
      self.assertEqual( gzoi_id, id( MyTemplatedMethodClass.__dict__[ 'GetSizeOL<int>' ] ) )

    # implicitly forced instantiation
      self.assertEqual( m.GetSizeOL( MyDoubleVector_t() ), m.GetVectorOfDoubleSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + 2 )
      for key in MyTemplatedMethodClass.__dict__.keys():
       # the actual method name is implementation dependent (due to the
       # default vars, and vector could live in a versioned namespace),
       # so find it explicitly:
         if key[0:9] == 'GetSizeOL' and 'vector<double' in key:
            mname = key
      self.assert_( mname in dir(MyTemplatedMethodClass) )
      gzoi_id = id( MyTemplatedMethodClass.__dict__[ mname ] )

    # as above, no changes on 2nd call
      self.assertEqual( m.GetSizeOL( MyDoubleVector_t() ), m.GetVectorOfDoubleSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + 2 )
      self.assertEqual( gzoi_id, id( MyTemplatedMethodClass.__dict__[ mname ] ) )

   def test7TemplateGlobalFunctions( self ):
      """Test template global function lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...

      f = MyTemplatedFunction

      self.assertEqual( f( 1 ), 1 )
      self.assertEqual( type( f( 2 ) ), type( 2 ) )
      self.assertEqual( f( 3. ), 3. )
      self.assertEqual( type( f( 4. ) ), type( 4. ) )

   def test8TemplatedArgument( self ):
      """Use of template argument"""

      obj = MyTemplateTypedef()
      obj.set( 'hi' )   # used to fail with TypeError


### C++ by-non-const-ref arguments tests =====================================
class Cpp03PassByNonConstRef( MyTestCase ):
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
class Cpp04HandlingAbstractClasses( MyTestCase ):
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
class Cpp05AssignToRefArbitraryClass( MyTestCase ):
   def test1AssignToReturnByRef( self ):
      """Test assignment to an instance returned by reference"""

      a = std.vector( RefTester )()
      a.push_back( RefTester( 42 ) )

      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 42 )

      a[0] = RefTester( 33 )
      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 33 )

   def test2NiceErrorMessageReturnByRef( self ):
      """Want nice error message of failing assign by reference"""

      a = RefTesterNoAssign()
      self.assertEqual( type(a), type(a[0]) )

      self.assertRaises( TypeError, a.__setitem__, 0, RefTesterNoAssign() )
      try:
         a[0] = RefTesterNoAssign()
      except TypeError, e:
         self.assert_( 'can not assign' in str(e) )


### Check availability of math conversions ===================================
class Cpp06MathConverters( MyTestCase ):
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
class Cpp07GloballyOverloadedComparator( MyTestCase ):
   def test1Comparator( self ):
      """Check that the global operator!=/== is picked up"""

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

   def test2Comparator( self ):
      """Check that the namespaced global operator!=/== is picked up"""

      a, b = ComparableSpace.NSComparable(), ComparableSpace.NSComparable()

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



### Check access to global variables =========================================
class Cpp08GlobalVariables( MyTestCase ):
   def test1DoubleArray( self ):
      """Verify access to array of doubles"""

      self.assertEqual( myGlobalDouble, 12. )
      self.assertRaises( IndexError, myGlobalArray.__getitem__, 500 )

   def test2WriteGlobalInstances( self ):
      """Verify writability of global instances"""

      import ROOT

      def verify( func, name, val ):
         self.assertEqual( func(),                    val )
         self.assertEqual( getattr( ROOT, name ),     val )

      verify( ROOT.PR_GetLumi1, "PR_Lumi1", "::1 C++ global lumi" )

      ROOT.PR_Lumi1 = "::1 python global lumi"
      verify( ROOT.PR_GetLumi1, "PR_Lumi1", "::1 python global lumi" )

      ROOT.PR_Lumi2 = "::2 python global lumi"
      verify( ROOT.PR_GetLumi2, "PR_Lumi2", "::2 python global lumi" )

      def verify( func, name, val ):
         self.assertEqual( func(),                      val )
         self.assertEqual( getattr( NS_PR_Lumi, name ), val )

      verify( NS_PR_Lumi.PR_GetLumi1, "PR_Lumi1", "NS::1 C++ global lumi" )

      NS_PR_Lumi.PR_Lumi1 = "NS::1 python global lumi"
      verify( NS_PR_Lumi.PR_GetLumi1, "PR_Lumi1", "NS::1 python global lumi" )

      NS_PR_Lumi.PR_Lumi2 = "NS::2 python global lumi"
      verify( NS_PR_Lumi.PR_GetLumi2, "PR_Lumi2", "NS::2 python global lumi" )


### Verify temporary handling for long expressions ===========================
class Cpp09LongExpressions( MyTestCase ):
   def test1LongExpressionWithTemporary( self ):
      """Test life time of temporary in long expression"""

      self.assertEqual( SomeClassWithData.SomeData.s_numData, 0 )
      r = SomeClassWithData()
      self.assertEqual( SomeClassWithData.SomeData.s_numData, 1 )

    # in this, GimeData() returns a datamember of the temporary result
    # from GimeCopy(); normal ref-counting would let it go too early
      self.assertEqual( r.GimeCopy().GimeData().s_numData, 2 )

      del r
      self.assertEqual( SomeClassWithData.SomeData.s_numData, 0 )


### Test usability of standard exceptions ====================================
class Cpp10StandardExceptions( MyTestCase ):
   def test1StandardExceptionsAccessFromPython( self ):
      """Access C++ standard exception objects from python"""

      e = std.runtime_error( "runtime pb!!" )
      self.assert_( e )
      self.assertEqual( e.what(), "runtime pb!!" )


### Test handing over of pointer variables from containers ===================
class Cpp11PointerContainers( MyTestCase ):
   def test1MapAssignment( self ):
      """Assignment operator on pointers"""

      def fill( self ):
         v = std.vector( 'PR_ValClass*' )()
         n = fillVect( v )

         self.assertEqual( n, len(v) )
         self.assertEqual( v[0].m_val, "aap" )

         m = {}
         for i in range( n ):
            m[i] = v[i]

         self.assertEqual( len(m), len(v) )
         return m

      m = fill( self )
      self.assertEqual( m[0].m_val, "aap" )


### Test lookup of lazily created functions in namespaces ====================
class Cpp12NamespaceLazyFunctions( MyTestCase ):
   def test1NamespaceLazyFunctions( self ):
      """Lazy lookup of late created functions"""

      import cppyy
      cppyy.gbl.gInterpreter.ProcessLine( 'namespace PyCpp12_ns_test1 {}' )
      cppyy.gbl.gInterpreter.ProcessLine(
         'namespace PyCpp12_ns_test1 { class PyCpp12_A {}; int PyCpp12_f() {return 32;}; }' )

      self.assert_( cppyy.gbl.PyCpp12_ns_test1.PyCpp12_A() )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test1.PyCpp12_f(), 32 )

   def test2NamespaceOverloadedLazyFunctions( self ):
      """Lazy lookup of late created overloaded functions"""

      import cppyy
      cppyy.gbl.gInterpreter.ProcessLine( 'namespace PyCpp12_ns_test2 {}')
      cppyy.gbl.gInterpreter.ProcessLine(
         'namespace PyCpp12_ns_test2 { class PyCpp12_A {}; \
          int PyCpp12_f(int n) {return 32*n;} \
          int PyCpp12_f() {return 32;}; }')

      self.assert_( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_A() )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_f(2), 64 )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_f(),  32 )

      cppyy.gbl.gInterpreter.ProcessLine(
         'namespace PyCpp12_ns_test2 { \
          int PyCpp12_g(const std::string&) {return 42;} \
          int PyCpp12_g() {return 13;}; }')

      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_g(''), 42 )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_g(),   13 )


### Usage of custom new/delete functions =====================================
class Cpp13OverloadedNewDelete( MyTestCase ):
   def test1StaticData( self ):
      """Use of custom new/delete"""

      import cppyy, gc

      self.assertEqual( cppyy.gbl.PR_StaticStuff(int).describe(), 'StaticStuff::s_data -> 999')
      m = cppyy.gbl.PR_CustomNewDeleteClass()
      self.assertEqual( cppyy.gbl.PR_StaticStuff(int).describe(), 'StaticStuff::s_data -> 123')
      del m; gc.collect()
      self.assertEqual( cppyy.gbl.PR_StaticStuff(int).describe(), 'StaticStuff::s_data -> 321')


### Copy constructor ordering determines overload ============================
class Cpp14CopyConstructorOrdering( MyTestCase ):
   def test1NoUserCCtor( self ):
      """Overload with implicit copy ctor"""

      import cppyy
      m1 = cppyy.gbl.MyCopyingClass1()
      m2 = cppyy.gbl.MyCopyingClass1(2, 2)

      self.assertEqual( float(m1), -2. )
      self.assertEqual( float(m2),  4. )

      m3 = cppyy.gbl.MyCopyingClass1( m2 )
      self.assertEqual( m3.m_d1, m2.m_d1 )
      self.assertEqual( m3.m_d2, m2.m_d2 )

   def test2NoUserCCtor( self ):
      """Overload with user provided cctor second"""

      import cppyy
      m1 = cppyy.gbl.MyCopyingClass2()
      m2 = cppyy.gbl.MyCopyingClass2(2, 2)

      self.assertEqual( float(m1), -2. )
      self.assertEqual( float(m2),  4. )

      m3 = cppyy.gbl.MyCopyingClass2( m2 )
      self.assertEqual( m3.m_d1, m2.m_d1 )
      self.assertEqual( m3.m_d2, m2.m_d2 )

   def test3NoUserCCtor( self ):
      """Overload with user provided cctor first"""

      import cppyy
      m1 = cppyy.gbl.MyCopyingClass3()
      m2 = cppyy.gbl.MyCopyingClass3(2, 2)

      self.assertEqual( float(m1), -2. )
      self.assertEqual( float(m2),  4. )

      m3 = cppyy.gbl.MyCopyingClass3( m2 )
      self.assertEqual( m3.m_d1, m2.m_d1 )
      self.assertEqual( m3.m_d2, m2.m_d2 )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
