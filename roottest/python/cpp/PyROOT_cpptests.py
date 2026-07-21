# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Author: Enric Tejedor Saavedra
# Author: Vincenzo Eduardo Padulano (CERN)
# Created: 01/03/05
# Last: 10/08/21 (MM-DD-YY)

"""C++ language interface unit tests for PyROOT package.

NOTE: several of the original test cases in this file were removed because
they are covered upstream in the cppyy test suite
(bindings/pyroot/cppyy/cppyy/test/): see test_datatypes.py (enums, object
validity, object/pointer comparisons), test_advancedcpp.py (namespaces, void
pointer passing), test_pythonify.py (underscore in class names),
test_regression.py and test_overloads.py (deep inheritance overload
resolution). What remains are ROOT-specific tests and tests without (green)
upstream equivalents.
"""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sys import maxsize

import ROOT
from ROOT import TObject, TLorentzVector, TVectorF, TROOT, gROOT, TMatrixD, TString, std
from ROOT import MakeNullPointer, AsCObject, BindObject, AddressOf, addressof
from common import *

__all__ = [
   'Cpp1LanguageFeatureTestCase',
   'Cpp2ClassNamingTestCase',
   'Cpp3UsingDeclarations',
   'Cpp4InheritanceTreeOverloadResolution',
]

### C++ language constructs test cases =======================================
class Cpp1LanguageFeatureTestCase( MyTestCase ):
   # NOTE: test04NsEnumType and test06ScopedEnum are kept although covered by
   # cppyy's test_datatypes.py::test12_enum_scopes, because that upstream test
   # is currently marked xfail.
   def test04NsEnumType(self):
      """Test lookup type of enum in namespace"""
      ROOT.gInterpreter.Declare("namespace myns { enum foo { aa,bb }; }")

      self.assertEqual(ROOT.myns.aa, 0)
      self.assertEqual(ROOT.myns.bb, 1)

      cppname = ROOT.myns.foo.__cpp_name__
      self.assertEqual(cppname, 'myns::foo')

      self.assertEqual(ROOT.myns.foo.aa, 0)
      self.assertEqual(ROOT.myns.foo.bb, 1)

   def test06ScopedEnum(self):
      """Test lookup of scoped enums and their values"""
      ROOT.gInterpreter.Declare("enum class scopedEnum { gg=1,hh };")
      self.assertEqual(ROOT.scopedEnum.gg, 1)

      ROOT.gInterpreter.Declare("namespace myns { enum class scopedEnum { gg=1,hh }; }")
      self.assertEqual(ROOT.myns.scopedEnum.gg, 1)

   def test07CopyContructor( self ):
      """Test copy constructor"""

      t1 = TLorentzVector( 1., 2., 3., -4. )
      t2 = TLorentzVector( 0., 0., 0.,  0. )
      t3 = TLorentzVector( t1 )

      self.assertEqual( t1, t3 )
      self.assertNotEqual( t1, t2 )

      for i in range(4):
         self.assertEqual( t1[i], t3[i] )

      # Test copy constructor with null pointer
      t4 = MakeNullPointer(TLorentzVector)
      t4.__init__(TLorentzVector(0, 1, 2, 3))
      t5 = MakeNullPointer(TLorentzVector)
      TLorentzVector.__init__(t5, TLorentzVector(0, 1, 2, 3))

      # Test __assign__ if the object already exists
      t6 = TLorentzVector(0, 0, 0, 0)
      t6.__assign__(TLorentzVector(0, 1, 2, 3))
      t7 = TLorentzVector(0, 0, 0, 0)
      TLorentzVector.__assign__(t7, TLorentzVector(0, 1, 2, 3))

      for i in range(4):
         self.assertEqual( t4[i], t5[i] )
         self.assertEqual( t6[i], t7[i] )

   def test09ElementAccess( self ):
      """Test access to elements in matrix and array objects."""

      n = 3
      v = TVectorF( n )
      m = TMatrixD( n, n )

      for i in range(n):
         self.assertEqual( v[i], 0.0 )

         for j in range(n):
            self.assertEqual( m[i][j], 0.0 )

   def test10StaticFunctionCall( self ):
      """Test call to static function."""

      c1 = TROOT.Class()
      self.assertTrue( not not c1 )

      c2 = gROOT.Class()

      self.assertIs( c1, c2 )

      old = gROOT.GetDirLevel()
      TROOT.SetDirLevel( 2 )
      self.assertEqual( 2, gROOT.GetDirLevel() )
      gROOT.SetDirLevel( old )

      old = TROOT.GetDirLevel()
      gROOT.SetDirLevel( 3 )
      self.assertEqual( 3, TROOT.GetDirLevel() )
      TROOT.SetDirLevel( old )

   def test13Macro( self ):
      """Test access to cpp macro's"""

      gROOT.ProcessLine( '#define aap "aap"' )
      gROOT.ProcessLine( '#define noot 1' )
      gROOT.ProcessLine( '#define mies 2.0' )

      # looking up macro's is slow, so needs to be explicit (note that NULL,
      # see above, is a special case)
      ROOT.PyConfig.ExposeCppMacros = True

      # test also that garbage macros are not found
      self.assertRaises( AttributeError, getattr, ROOT, "_this_does_not_exist_at_all" )

      ROOT.PyConfig.ExposeCppMacros = False

   def test14OpaquePointerPassing( self ):
      """Test passing around of opaque pointers"""

      import ROOT

      s = TString( "Hello World!" )
      co = AsCObject( s )

      ad = addressof( s )

      self.assertTrue( s == BindObject( co, s.__class__ ) )
      self.assertTrue( s == BindObject( co, "TString" ) )
      self.assertTrue( s == BindObject( ad, s.__class__ ) )
      self.assertTrue( s == BindObject( ad, "TString" ) )

   def test16AddressOfaddressof(self):
      """Test addresses returned by AddressOf and addressof"""
      import ROOT

      o = ROOT.TObject()

      addr_as_int    = ROOT.addressof(o)
      addr_as_buffer = ROOT.AddressOf(o)

      # The result of AddressOf can be passed to a function that expects a void*
      # or an integer pointer (Long64_t* for 64bit, Int_t* for 32bit)
      is_64bit = maxsize > 2**32
      if is_64bit:
         ROOT.gInterpreter.Declare("""
         Long64_t get_address_in_buffer_vp(void *p) { return *(Long64_t*)p; }
         Long64_t get_address_in_buffer_ip(Long64_t *p) { return *p; }
         """)
      else:
         ROOT.gInterpreter.Declare("""
         Int_t get_address_in_buffer_vp(void *p) { return *(Int_t*)p; }
         Int_t get_address_in_buffer_ip(Int_t *p) { return *p; }
         """)
      self.assertEqual(addr_as_int, ROOT.get_address_in_buffer_vp(addr_as_buffer))
      self.assertEqual(addr_as_int, ROOT.get_address_in_buffer_ip(addr_as_buffer))
      self.assertEqual(addr_as_int, addr_as_buffer[0])


### C++ language naming of classes ===========================================
class Cpp2ClassNamingTestCase( MyTestCase ):
   def test03NamespaceInTemplates( self ):
      """Templated data members need to retain namespaces of arguments"""

      gROOT.LoadMacro( "Namespace.C+" )
      PR_NS_A = ROOT.PR_NS_A

      p = std.pair( std.vector( PR_NS_A.PR_ST_B ), std.vector( PR_NS_A.PR_NS_D.PR_ST_E ) )()
      self.assertTrue( "vector<PR_NS_A::PR_ST_B>" in type(p.first).__name__ )
      self.assertTrue( "vector<PR_NS_A::PR_NS_D::PR_ST_E>" in type(p.second).__name__ )


### C++ language using declarations ===========================================
class Cpp3UsingDeclarations( MyTestCase ):
    def test1TGraphMultiErrorsSetLineColor(self):
        """Test that a using function declaration is picked up by the overload resolution"""

        # This line breaks with the following error if the using function declaration is not picked up
        # TypeError: void TGraphMultiErrors::SetLineColor(int e, short lcolor) =>
        #   TypeError: takes at least 2 arguments (1 given)
        ROOT.TGraphMultiErrors().SetLineColor(0)

    def test2TH1FConstructor(self):
        """Test that the using declaration of a constructor is picked up by the overload resolution"""
        # ROOT-10786

        ROOT.gInterpreter.Declare("""
        class MyTH1F : public TH1F {
        public:
            using TH1F::TH1F;
        };
        """)

        h = ROOT.MyTH1F("name", "title", 100, 0, 100)


class Cpp4InheritanceTreeOverloadResolution(MyTestCase):
    """
    Tests correct overload resolution of functions accepting classes part of
    the same inheritance tree.
    """

    @classmethod
    def setUpClass(cls):
        """Declare the classes and functions needed for the tests"""
        ROOT.gInterpreter.Declare(
        """
        namespace Cpp4 {
            class A {};
            class B: public A {};
            class C: public B {};

            class X {};
            class Y: public X {};
            class Z: public Y {};

            int myfunc(const B &b){
                return 1;
            }

            int myfunc(const C &c){
                return 2;
            }

            int myfunc(const B &b, const Z &z){
                return 1;
            }

            int myfunc(const C &c, const X &x){
                return 2;
            }

        } // end namespace
        """)

    def test2TwoArgumentFunctionAmbiguous(self):
        """
        Test the behaviour of a scenario that would be ambiguous in C++.

        In PyROOT, the function with the highest priority in the overload
        resolution will be called. Would be nice to throw an error in this kind
        of scenario.
        """
        # In C++ calling myfunc(C(), Z()) would throw
        # error: call to 'myfunc' is ambiguous
        self.assertEqual(ROOT.Cpp4.myfunc(ROOT.Cpp4.C(), ROOT.Cpp4.Z()), 1)


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
