# File: roottest/python/cpp/PyROOT_advancedtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 06/04/05
# Last: 04/27/16

"""C++ advanced language interface unit tests for PyROOT package.

NOTE: most of the original test cases in this file were removed because they
are covered upstream in the cppyy test suite (bindings/pyroot/cppyy/cppyy/test/),
which descends from the same original PyROOT tests: see test_advancedcpp.py,
test_templates.py, test_lowlevel.py, test_operators.py and test_stltypes.py.
What remains are tests without (green) upstream equivalents.
"""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ROOT
from ROOT import gROOT
from common import *

__all__ = [
   'Cpp05AssignToRefArbitraryClass',
   'Cpp07GloballyOverloadedComparator',
   'Cpp08GlobalVariables',
   'Cpp10StandardExceptions',
   'Cpp12NamespaceLazyFunctions',
   'Cpp14CopyConstructorOrdering',
]

gROOT.LoadMacro( "AdvancedCpp.C+" )


### Return by reference should call assignment operator ======================
class Cpp05AssignToRefArbitraryClass( MyTestCase ):
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


### Check global operator== overload =========================================
class Cpp07GloballyOverloadedComparator( MyTestCase ):
   def test3DirectUseComparator( self ):
      """Check that a namespaced global operator!=/== can be used directly"""

      ComparableSpace = ROOT.ComparableSpace

      eq = getattr(ComparableSpace, 'operator==')
      ComparableSpace.NSComparable.__eq__ = eq

      a, b = ComparableSpace.NSComparable(), ComparableSpace.NSComparable()

      self.assertEqual( a, b )
      self.assertEqual( b, a )


### Check access to global variables =========================================
# NOTE: kept although covered by cppyy's
# test_advancedcpp.py::test21_access_to_global_variables, because that
# upstream test is currently marked xfail.
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


### Test usability of standard exceptions ====================================
class Cpp10StandardExceptions( MyTestCase ):
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


### Test lookup of lazily created functions in namespaces ====================
class Cpp12NamespaceLazyFunctions( MyTestCase ):
   def test1NamespaceLazyFunctions( self ):
      """Lazy lookup of late created functions"""

      import cppyy
      cppyy.cppdef( 'namespace PyCpp12_ns_test1 {}' )
      cppyy.cppdef(
         'namespace PyCpp12_ns_test1 { class PyCpp12_A {}; int PyCpp12_f() {return 32;}; }' )

      self.assertTrue( cppyy.gbl.PyCpp12_ns_test1.PyCpp12_A() )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test1.PyCpp12_f(), 32 )

   def test2NamespaceOverloadedLazyFunctions( self ):
      """Lazy lookup of late created overloaded functions"""

      import cppyy
      cppyy.cppdef( 'namespace PyCpp12_ns_test2 {}')
      cppyy.cppdef(
         'namespace PyCpp12_ns_test2 { class PyCpp12_A {}; \
          int PyCpp12_f(int n) {return 32*n;} \
          int PyCpp12_f() {return 32;}; }')

      self.assertTrue( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_A() )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_f(2), 64 )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_f(),  32 )

      cppyy.cppdef(
         'namespace PyCpp12_ns_test2 { \
          int PyCpp12_g(const std::string&) {return 42;} \
          int PyCpp12_g() {return 13;}; }')

      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_g(''), 42 )
      self.assertEqual( cppyy.gbl.PyCpp12_ns_test2.PyCpp12_g(),   13 )


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
