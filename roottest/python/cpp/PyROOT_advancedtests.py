# File: roottest/python/cpp/PyROOT_advancedtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 04/27/16

"""C++ advanced language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ROOT
from ROOT import gROOT, std
from common import *

import ctypes

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


PR_B = ROOT.PR_B
PR_C = ROOT.PR_C
PR_D = ROOT.PR_D
GetA = ROOT.GetA
GetB = ROOT.GetB
GetC = ROOT.GetC
GetD = ROOT.GetD
T1 = ROOT.T1
T2 = ROOT.T2


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
   def test01SingleInstantiatedTemplate( self ):
      """Test data member access for a templated class"""

      t1 = T1( int )( 32 )
      self.assertEqual( t1.value(), 32 )
      self.assertEqual( t1.m_t1, 32 )

      t1.m_t1 = 41
      self.assertEqual( t1.value(), 41 )
      self.assertEqual( t1.m_t1, 41 )

   def test02TemplateInstantiatedTemplate( self ):
      """Test data member access for a templated class instantiated with a template"""

      t2 = T2( T1( int ) )()
      t2.m_t2.m_t1 = 32
      self.assertEqual( t2.m_t2.value(), 32 )
      self.assertEqual( t2.m_t2.m_t1, 32 )

   def test03TemplateInstantiationWithVectorOfFloat( self ):
      """Test template instantiation with a std::vector< float >"""

      gROOT.LoadMacro( "Template.C+" )
      MyTemplatedClass = ROOT.MyTemplatedClass


    # the following will simply fail if there is a naming problem (e.g. std::,
    # allocator<int>, etc., etc.); note the parsing required ...
      b = MyTemplatedClass( std.vector( float ) )()

      for i in range(5):
         b.m_b.push_back( i )
         self.assertEqual( round( b.m_b[i], 5 ), float(i) )

   def test04TemplateMemberFunctions( self ):
      """Test template member functions lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...
      MyTemplatedMethodClass = ROOT.MyTemplatedMethodClass

      m = MyTemplatedMethodClass()

    # template without arguments can not resolve; check that there is an exception
    # and a descriptive error message
      self.assertRaises( TypeError, m.GetSize )
      try:
         m.GetSize()
      except TypeError as e:
         self.assertTrue( "Template method resolution failed" in str(e) )

      # New cppyy needs square brackets for explicit instantiation here,
      # otherwise it tries to call the template proxy with the passed
      # argument and it fails, since no instantiation is available.
      self.assertEqual( m.GetSize['char'](),   m.GetCharSize() )
      self.assertEqual( m.GetSize[int](),      m.GetIntSize() )
      self.assertEqual( m.GetSize['long'](),   m.GetLongSize() )
      self.assertEqual( m.GetSize[float](),    m.GetFloatSize() )
      self.assertEqual( m.GetSize['double'](), m.GetDoubleSize() )

      self.assertEqual( m.GetSize['MyDoubleVector_t'](), m.GetVectorOfDoubleSize() )
      self.assertEqual( m.GetSize['vector<double>'](), m.GetVectorOfDoubleSize() )

   def test05TemplateMemberFunctions( self ):
      """Test template member functions lookup and calls (set 2)"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...
      MyTemplatedMethodClass = ROOT.MyTemplatedMethodClass

      m = MyTemplatedMethodClass()

    # note that the function and template arguments are reverted
      self.assertRaises( TypeError, m.GetSize2( 'char', 'long' ), 'a', 1 )
      # In the new Cppyy, we need to use square brackets in this case for
      # the bindings to know we are explicitly instantiating for char,long.
      # Otherwise, the templated parameters are just (mis)interpreted as
      # strings and a call to the string,string instantiation is made.
      self.assertEqual(m.GetSize2['char', 'long']( 1, 'a' ), m.GetCharSize() - m.GetLongSize() )

      # Cppyy's Long will be deprecated in favour of ctypes.c_long
      # https://bitbucket.org/wlav/cppyy/issues/101
      long_par = ctypes.c_long(256).value
      self.assertEqual( m.GetSize2(long_par, 1.), m.GetDoubleSize() - m.GetIntSize() )

   def test06OverloadedTemplateMemberFunctions( self ):
      """Test overloaded template member functions lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...
      MyTemplatedMethodClass = ROOT.MyTemplatedMethodClass
      MyDoubleVector_t = ROOT.MyDoubleVector_t

      m = MyTemplatedMethodClass()

    # the number of entries in the class dir() is used to check whether
    # member templates have been instantiated on it
      nd = len(dir(MyTemplatedMethodClass))

    # use existing (note '-' to make sure the correct call was made
      self.assertEqual( m.GetSizeOL( 1 ),             -m.GetLongSize() )
      self.assertEqual( m.GetSizeOL( "aapje" ),       -len("aapje") )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd )

    # use existing explicit instantiations
      # New cppyy: use bracket syntax for explicit instantiation
      self.assertEqual( m.GetSizeOL[float]( 3.14 ),  m.GetFloatSize() )
      self.assertEqual( m.GetSizeOL( 3.14 ), m.GetDoubleSize() )
      num_new_inst = 2

      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + num_new_inst)

    # explicit forced instantiation
      # New cppyy: use bracket syntax for explicit instantiation
      inst = m.GetSizeOL[int]
      self.assertEqual( inst( 1 ),       m.GetIntSize() )
      num_new_inst += 1
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + num_new_inst )
      self.assertTrue( 'GetSizeOL<int>' in dir(MyTemplatedMethodClass) )
      gzoi_id = id( MyTemplatedMethodClass.__dict__[ 'GetSizeOL<int>' ] )

    # second call should make no changes, but re-use
      self.assertEqual( inst( 1 ),       m.GetIntSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + num_new_inst )
      self.assertEqual( gzoi_id, id( MyTemplatedMethodClass.__dict__[ 'GetSizeOL<int>' ] ) )

    # implicitly forced instantiation
      self.assertEqual( m.GetSizeOL( MyDoubleVector_t() ), m.GetVectorOfDoubleSize() )
      num_new_inst += 1
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + num_new_inst )
      for key in MyTemplatedMethodClass.__dict__.keys():
       # the actual method name is implementation dependent (due to the
       # default vars, and vector could live in a versioned namespace),
       # so find it explicitly:
         if key[0:9] == 'GetSizeOL' and 'vector<double' in key:
            mname = key
      self.assertTrue( mname in dir(MyTemplatedMethodClass) )
      gzoi_id = id( MyTemplatedMethodClass.__dict__[ mname ] )

    # as above, no changes on 2nd call
      self.assertEqual( m.GetSizeOL( MyDoubleVector_t() ), m.GetVectorOfDoubleSize() )
      self.assertEqual( len(dir(MyTemplatedMethodClass)), nd + num_new_inst )
      self.assertEqual( gzoi_id, id( MyTemplatedMethodClass.__dict__[ mname ] ) )

   def test07TemplateMemberFunctionsNotInstantiated(self):
      """Test lookup and calls for template member functions
      that have not been explicitly instantiated"""
      MyTemplatedMethodClass = ROOT.MyTemplatedMethodClass
      m = MyTemplatedMethodClass()

      # Test the templated overload
      # In the new Cppyy, we need to use square brackets in this case for
      # the bindings to know we are explicitly instantiating for char.
      # Otherwise, the templated parameter is just (mis)interpreted as
      # string and a call to the string instantiation is made.
      self.assertEqual(m.GetSizeNEI['char']('c'), m.GetCharSize())
      # This instantiation also needs square brackets in new Cppyy
      self.assertEqual(m.GetSizeNEI[int](1), m.GetIntSize())

      # Test the non-templated overload (must have been added to
      # the template proxy too)
      self.assertEqual(m.GetSizeNEI(), 1)

   def test08TemplateGlobalFunctions( self ):
      """Test template global function lookup and calls"""

    # gROOT.LoadMacro( "Template.C+" )  # already loaded ...
      MyTemplatedFunction = ROOT.MyTemplatedFunction

      f = MyTemplatedFunction

      self.assertEqual( f( 1 ), 1 )
      self.assertEqual( type( f( 2 ) ), type( 2 ) )
      self.assertEqual( f( 3. ), 3. )
      self.assertEqual( type( f( 4. ) ), type( 4. ) )

   def test09TemplatedArgument( self ):
      """Use of template argument"""

      obj = ROOT.MyTemplateTypedef()
      obj.set( 'hi' )   # used to fail with TypeError

   def test10TemplatedFunctionNamespace(self):
      """Test template function in a namespace, lookup and calls"""

      f = ROOT.MyNamespace.MyTemplatedFunctionNamespace

      val = 1.0
      v = ROOT.std.vector("float")()
      v.push_back(val)

      inst_float = f["float"]
      inst_float_t = f["Float_t"]
      inst_vec_float = f["vector<float>"]
      inst_std_vec_float = f["std::vector<float>"]

      # Test basic type
      self.assertEqual(inst_float(val), val)
      self.assertEqual(type(inst_float(val)), type(val))

      # Test typedef resolution
      self.assertEqual(inst_float_t(val), val)
      self.assertEqual(type(inst_float_t(val)), type(val))

      # Test no namespace specification
      self.assertEqual(inst_vec_float(v)[0], val)
      self.assertEqual(type(inst_vec_float(v)), type(v))

      # Test incomplete type specification
      # Complete type is std::vector<float, std::allocator<float>>
      self.assertEqual(inst_std_vec_float(v)[0], val)
      self.assertEqual(type(inst_std_vec_float(v)), type(v))

   def test11VariadicTemplates(self):
      """Test variadic templates resolution"""

      ROOT.gInterpreter.ProcessLine("""
      template<typename... MyTypes>
      int f() {return sizeof...(MyTypes);}
      """)

      res = ROOT.f['int', 'double', 'void*']()
      self.assertEqual(res, 3)


### C++ by-non-const-ref arguments tests =====================================
class Cpp03PassByNonConstRef( MyTestCase ):

   def test2PassBuiltinsByNonConstRef( self ):
      """Test parameter passing of builtins through non-const reference"""

      SetLongThroughRef = ROOT.SetLongThroughRef
      SetDoubleThroughRef = ROOT.SetDoubleThroughRef
      SetIntThroughRef = ROOT.SetIntThroughRef

      import ctypes
      l = ctypes.c_long(42)
      SetLongThroughRef( l, 41 )
      self.assertEqual( l.value, 41 )

      i = ctypes.c_int(42)
      SetIntThroughRef( i, 13 )
      self.assertEqual( i.value, 13 )

   def test3PassBuiltinsByNonConstRef( self ):
      """Test parameter passing of builtins through const reference"""

      self.assertEqual( ROOT.PassLongThroughConstRef( 42 ), 42 )
      self.assertEqual( ROOT.PassDoubleThroughConstRef( 3.1415 ), 3.1415 )
      self.assertEqual( ROOT.PassIntThroughConstRef( 42 ), 42 )


### C++ abstract classes should behave normally, but be non-instatiatable ====
class Cpp04HandlingAbstractClasses( MyTestCase ):
   def test1ClassHierarchy( self ):
      """Test abstract class in a hierarchy"""

      self.assertTrue( issubclass( ROOT.MyConcreteClass, ROOT.MyAbstractClass ) )

      c = ROOT.MyConcreteClass()
      self.assertTrue( isinstance( c, ROOT.MyConcreteClass ) )
      self.assertTrue( isinstance( c, ROOT.MyAbstractClass ) )

   def test2Instantiation( self ):
      """Test non-instatiatability of abstract classes"""

      self.assertRaises( TypeError, ROOT.MyAbstractClass )


### Return by reference should call assignment operator ======================
class Cpp05AssignToRefArbitraryClass( MyTestCase ):
   def test1AssignToReturnByRef( self ):
      """Test assignment to an instance returned by reference"""

      RefTester = ROOT.RefTester

      a = std.vector( RefTester )()
      a.push_back( RefTester( 42 ) )

      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 42 )

      a[0] = RefTester( 33 )
      self.assertEqual( len(a), 1 )
      self.assertEqual( a[0].m_i, 33 )

   def test2NiceErrorMessageReturnByRef( self ):
      """Want nice error message of failing assign by reference"""

      RefTesterNoAssign = ROOT.RefTesterNoAssign

      a = RefTesterNoAssign()
      self.assertEqual( type(a), type(a[0]) )

      self.assertRaises( TypeError, a.__setitem__, 0, RefTesterNoAssign() )
      try:
         a[0] = RefTesterNoAssign()
      except TypeError as e:
         self.assertTrue( 'cannot assign' in str(e) )


### Check availability of math conversions ===================================
class Cpp06MathConverters( MyTestCase ):
   def test1MathConverters( self ):
      """Test operator int/long/double incl. typedef"""

      a = ROOT.Convertible()
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

      a, b = ROOT.Comparable(), ROOT.Comparable()

      self.assertEqual( a, b )
      self.assertEqual( b, a )
      self.assertTrue( a.__eq__( b ) )
      self.assertTrue( b.__eq__( a ) )
      self.assertTrue( a.__ne__( a ) )
      self.assertTrue( b.__ne__( b ) )
      self.assertEqual( a.__eq__( b ), True )
      self.assertEqual( b.__eq__( a ), True )
      self.assertEqual( a.__eq__( a ), False )
      self.assertEqual( b.__eq__( b ), False )

   def test2Comparator( self ):
      """Check that the namespaced global operator!=/== is picked up"""

      a, b = ROOT.ComparableSpace.NSComparable(), ROOT.ComparableSpace.NSComparable()

      self.assertEqual( a, b )
      self.assertEqual( b, a )
      self.assertTrue( a.__eq__( b ) )
      self.assertTrue( b.__eq__( a ) )
      self.assertTrue( a.__ne__( a ) )
      self.assertTrue( b.__ne__( b ) )
      self.assertEqual( a.__eq__( b ), True )
      self.assertEqual( b.__eq__( a ), True )
      self.assertEqual( a.__eq__( a ), False )
      self.assertEqual( b.__eq__( b ), False )

   def test3DirectUseComparator( self ):
      """Check that a namespaced global operator!=/== can be used directly"""

      ComparableSpace = ROOT.ComparableSpace

      eq = getattr(ComparableSpace, 'operator==')
      ComparableSpace.NSComparable.__eq__ = eq

      a, b = ComparableSpace.NSComparable(), ComparableSpace.NSComparable()

      self.assertEqual( a, b )
      self.assertEqual( b, a )


### Check access to global variables =========================================
class Cpp08GlobalVariables( MyTestCase ):
   def test1DoubleArray( self ):
      """Verify access to array of doubles"""

      self.assertEqual( ROOT.myGlobalDouble, 12. )
      self.assertRaises( IndexError, ROOT.myGlobalArray.__getitem__, 500 )

   def test2WriteGlobalInstances( self ):
      """Verify writability of global instances"""

      NS_PR_Lumi = ROOT.NS_PR_Lumi

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

      SomeClassWithData = ROOT.SomeClassWithData

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
      self.assertTrue( e )
      self.assertEqual( e.what(), "runtime pb!!" )

   def test2ExceptionBoolValue(self):
      """Test boolean value of exception object"""
      # ROOT-10870

      ROOT.gInterpreter.Declare("""
      namespace test2Exception {
         template <typename T>
         class Handle;

         class Exception : public std::exception {};
      }

      template <typename T>
      class test2Exception::Handle {
      public:
         std::shared_ptr<test2Exception::Exception const> returnsNull() const noexcept;
         std::shared_ptr<test2Exception::Exception const> returnsNotNull() const noexcept;
      };

      template<class T>
      inline std::shared_ptr<test2Exception::Exception const>
      test2Exception::Handle<T>::returnsNull() const noexcept { return std::shared_ptr<test2Exception::Exception>(); }

      template<class T>
      inline std::shared_ptr<test2Exception::Exception const>
      test2Exception::Handle<T>::returnsNotNull() const noexcept { return std::shared_ptr<test2Exception::Exception>(new test2Exception::Exception()); }
      """)

      handle = ROOT.test2Exception.Handle('int')()

      self.assertTrue(not handle.returnsNull());
      self.assertTrue(handle.returnsNotNull);


### Test handing over of pointer variables from containers ===================
class Cpp11PointerContainers( MyTestCase ):
   def test1MapAssignment( self ):
      """Assignment operator on pointers"""
      fillVect = ROOT.fillVect

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

      self.assertTrue( cppyy.gbl.PyCpp12_ns_test1.PyCpp12_A() )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test1.PyCpp12_f(), 32 )

   def test2NamespaceOverloadedLazyFunctions( self ):
      """Lazy lookup of late created overloaded functions"""

      import cppyy
      cppyy.gbl.gInterpreter.ProcessLine( 'namespace PyCpp12_ns_test2 {}')
      cppyy.gbl.gInterpreter.ProcessLine(
         'namespace PyCpp12_ns_test2 { class PyCpp12_A {}; \
          int PyCpp12_f(int n) {return 32*n;} \
          int PyCpp12_f() {return 32;}; }')

      self.assertTrue( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_A() )
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
