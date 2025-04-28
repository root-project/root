# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Author: Enric Tejedor Saavedra
# Author: Vincenzo Eduardo Padulano (CERN)
# Created: 01/03/05
# Last: 10/08/21 (MM-DD-YY)

"""C++ language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sys import maxsize

import ROOT
from ROOT import TObject, TLorentzVector, kRed, kGreen, kBlue, TVectorF, TROOT, TCanvas, gInterpreter, gROOT, TMatrixD, TString, std
from ROOT import MakeNullPointer, AsCObject, BindObject, AddressOf, addressof
from common import *
from functools import partial

__all__ = [
   'Cpp1LanguageFeatureTestCase',
   'Cpp2ClassNamingTestCase',
   'Cpp3UsingDeclarations',
   'Cpp4InheritanceTreeOverloadResolution',
]

IS_WINDOWS = 0
if 'win32' in sys.platform:
    import platform
    if '64' in platform.architecture()[0]:
        IS_WINDOWS = 64
    else:
        IS_WINDOWS = 32

### C++ language constructs test cases =======================================
class Cpp1LanguageFeatureTestCase( MyTestCase ):
   def test01ClassEnum( self ):
      """Test class enum access and values"""

      self.assertEqual( TObject.kBitMask,    gROOT.ProcessLine( "return TObject::kBitMask;" ) )
      self.assertEqual( TObject.kIsOnHeap,   gROOT.ProcessLine( "return TObject::kIsOnHeap;" ) )
      self.assertEqual( TObject.kNotDeleted, gROOT.ProcessLine( "return TObject::kNotDeleted;" ) )
      self.assertEqual( TObject.kZombie,     gROOT.ProcessLine( "return TObject::kZombie;" ) )

      t = TObject()

      self.assertEqual( TObject.kBitMask,    t.kBitMask )
      self.assertEqual( TObject.kIsOnHeap,   t.kIsOnHeap )
      self.assertEqual( TObject.kNotDeleted, t.kNotDeleted )
      self.assertEqual( TObject.kZombie,     t.kZombie )

   def test02Globalenum( self ):
      """Test global enums access and values"""

      self.assertEqual( kRed,   gROOT.ProcessLine( "return kRed;" ) )
      self.assertEqual( kGreen, gROOT.ProcessLine( "return kGreen;" ) )
      self.assertEqual( kBlue,  gROOT.ProcessLine( "return kBlue;" ) )

   def test03GlobalEnumType(self):
      """Test lookup and type of global enum"""
      ROOT.gInterpreter.Declare("enum foo { aa,bb };")

      self.assertEqual(ROOT.aa, 0)
      self.assertEqual(ROOT.bb, 1)

      cppname = ROOT.foo.__cpp_name__
      self.assertEqual(cppname, 'foo')

      self.assertEqual(ROOT.foo.aa, 0)
      self.assertEqual(ROOT.foo.bb, 1)

   def test04NsEnumType(self):
      """Test lookup type of enum in namespace"""
      ROOT.gInterpreter.Declare("namespace myns { enum foo { aa,bb }; }")

      self.assertEqual(ROOT.myns.aa, 0)
      self.assertEqual(ROOT.myns.bb, 1)

      cppname = ROOT.myns.foo.__cpp_name__
      self.assertEqual(cppname, 'myns::foo')

      self.assertEqual(ROOT.myns.foo.aa, 0)
      self.assertEqual(ROOT.myns.foo.bb, 1)

   def test05EnumSignedUnsigned(self):
      """Test lookup of enums with signed & unsigned underlying types"""
      ROOT.gInterpreter.Declare("enum bar { cc=-10,dd };")
      self.assertEqual(ROOT.cc, -10)

      ROOT.gInterpreter.Declare("enum bar2 { ee=4294967286,ff };")
      self.assertEqual(ROOT.ee, 4294967286)

      ROOT.gInterpreter.Declare("namespace myns { enum bar { cc=-10,dd }; }")
      self.assertEqual(ROOT.myns.cc, -10)

      ROOT.gInterpreter.Declare("namespace myns { enum bar2 { ee=4294967286,ff }; }")
      self.assertEqual(ROOT.myns.ee, 4294967286)

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

   def test08ObjectValidity( self ):
      """Test object validity checking"""

      t1 = TObject()

      self.assertTrue( t1 )
      self.assertTrue( not not t1 )

      t2 = gROOT.FindObject( "Nah, I don't exist" )

      self.assertTrue( not t2 )

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

      self.assertEqual( c1, c2 )

      old = gROOT.GetDirLevel()
      TROOT.SetDirLevel( 2 )
      self.assertEqual( 2, gROOT.GetDirLevel() )
      gROOT.SetDirLevel( old )

      old = TROOT.GetDirLevel()
      gROOT.SetDirLevel( 3 )
      self.assertEqual( 3, TROOT.GetDirLevel() )
      TROOT.SetDirLevel( old )

   def test11Namespaces( self ):
      """Test access to namespaces and inner classes"""

      gROOT.LoadMacro( "Namespace.C+" )
      PR_NS_A = ROOT.PR_NS_A

      self.assertEqual( PR_NS_A.sa,                            1 )
      self.assertEqual( PR_NS_A.PR_ST_B.sb,                    2 )
      self.assertEqual( PR_NS_A.PR_ST_B().fb,                 -2 )
      self.assertEqual( PR_NS_A.PR_ST_B.PR_ST_C.sc,            3 )
      self.assertEqual( PR_NS_A.PR_ST_B.PR_ST_C().fc,         -3 )
      self.assertEqual( PR_NS_A.PR_NS_D.sd,                    4 )
      self.assertEqual( PR_NS_A.PR_NS_D.PR_ST_E.se,            5 )
      self.assertEqual( PR_NS_A.PR_NS_D.PR_ST_E().fe,         -5 )
      self.assertEqual( PR_NS_A.PR_NS_D.PR_ST_E.PR_ST_F.sf,    6 )
      self.assertEqual( PR_NS_A.PR_NS_D.PR_ST_E.PR_ST_F().ff, -6 )

    # a few more, with namespaced typedefs
      self.assertEqual( PR_NS_A.tsa,                          -1 )
      self.assertEqual( PR_NS_A.ctsa,                         -1 )

    # data members coming in from a different namespace block
      self.assertEqual( PR_NS_A.tsa2,                         -1 )
      self.assertEqual( PR_NS_A.ctsa2,                        -1 )

    # data members from a different namespace in a separate file
      self.assertRaises( AttributeError, getattr, PR_NS_A, 'tsa3' )
      self.assertRaises( AttributeError, getattr, PR_NS_A, 'ctsa3' )

      gROOT.LoadMacro( "Namespace2.C+" )
      self.assertEqual( PR_NS_A.tsa3,                         -8 )
      self.assertEqual( PR_NS_A.ctsa3,                        -9 )

    # test equality of different lookup methods
      self.assertEqual( getattr( PR_NS_A, "PR_ST_B::PR_ST_C" ), PR_NS_A.PR_ST_B.PR_ST_C )
      self.assertEqual( getattr( PR_NS_A.PR_ST_B,  "PR_ST_C" ), PR_NS_A.PR_ST_B.PR_ST_C )

   def test12VoidPointerPassing( self ):
      """Test passing of variants of void pointer arguments"""

      gROOT.LoadMacro( "PointerPassing.C+" )
      
      Z = ROOT.Z

      o = TObject()
      oaddr = addressof(o)

      self.assertEqual( oaddr, Z.GimeAddressPtr( o ) )
      self.assertEqual( oaddr, Z.GimeAddressPtrRef( o ) )
      
      pZ = Z.getZ(0)
      self.assertEqual( Z.checkAddressOfZ( pZ ), True )
      self.assertEqual( pZ , Z.getZ(1) )

      import array
      # Not supported in p2.2
      # and no 8-byte integer type array on Windows 64b
      if hasattr( array.array, 'buffer_info' ) and IS_WINDOWS != 64:
         # New cppyy uses unsigned long to represent void* returns, as in DynamicCast.
         # To prevent an overflow error when converting the Python integer returned by
         # DynamicCast into a 4-byte signed long in 32 bits, we use unsigned long ('L')
         # as type of the array.array.
         array_t = 'L'
         addressofo = array.array( array_t, [o.IsA()._TClass__DynamicCast( o.IsA(), o )[0]] )
         self.assertEqual( addressofo.buffer_info()[0], Z.GimeAddressPtrPtr( addressofo ) )

      self.assertEqual( 0, Z.GimeAddressPtr( 0 ) );
      self.assertEqual( 0, Z.GimeAddressObject( 0 ) );

      ptr = MakeNullPointer( TObject )
      # New Cppyy does not raise ValueError,
      # it just returns zero
      self.assertEqual(addressof(ptr), 0)
      Z.SetAddressPtrRef( ptr )

      self.assertEqual( addressof( ptr ), 0x1234 )
      Z.SetAddressPtrPtr( ptr )
      self.assertEqual( addressof( ptr ), 0x4321 )

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

   def test15ObjectAndPointerComparisons( self ):
      """Verify object and pointer comparisons"""

      c1 = MakeNullPointer( TCanvas )
      self.assertEqual( c1, None )
      self.assertEqual( None, c1 )

      c2 = MakeNullPointer( TCanvas )
      self.assertEqual( c1, c2 )
      self.assertEqual( c2, c1 )

    # TLorentzVector overrides operator==
      l1 = MakeNullPointer( TLorentzVector )
      self.assertEqual( l1, None )
      self.assertEqual( None, l1 )

      self.assertNotEqual( c1, l1 )
      self.assertNotEqual( l1, c1 )

      l2 = MakeNullPointer( TLorentzVector )
      self.assertEqual( l1, l2 )
      self.assertEqual( l2, l1 )

      l3 = TLorentzVector( 1, 2, 3, 4 )
      l4 = TLorentzVector( 1, 2, 3, 4 )
      l5 = TLorentzVector( 4, 3, 2, 1 )
      self.assertEqual( l3, l4 )
      self.assertEqual( l4, l3 )

      self.assertTrue( l3 != None )        # like this to ensure __ne__ is called
      self.assertTrue( None != l3 )        # id.
      self.assertNotEqual( l3, l5 )
      self.assertNotEqual( l5, l3 )

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
   def test01Underscore( self ):
      """Test recognition of '_' as part of a valid class name"""

      z = ROOT.Z_()

      self.assertTrue( hasattr( z, 'myint' ) )
      self.assertTrue( z.GimeZ_( z ) )

   def test02DefaultCtorInNamespace( self ):
      """Check that constructor with default argument is found in namespace"""
      PR_NS_A = ROOT.PR_NS_A
      CtorWithDefaultInGBL = ROOT.CtorWithDefaultInGBL
      
      a = CtorWithDefaultInGBL()
      self.assertEqual( a.data, -1 )

      b = CtorWithDefaultInGBL( 1 )
      self.assertEqual( b.data, 1 )

      c = PR_NS_A.CtorWithDefaultInNS()
      self.assertEqual( c.data, -1 )

      c = PR_NS_A.CtorWithDefaultInNS( 2 )
      self.assertEqual( c.data, 2 )

   def test03NamespaceInTemplates( self ):
      """Templated data members need to retain namespaces of arguments"""

      PR_NS_A = ROOT.PR_NS_A

      p = std.pair( std.vector( PR_NS_A.PR_ST_B ), std.vector( PR_NS_A.PR_NS_D.PR_ST_E ) )()
      self.assertTrue( "vector<PR_NS_A::PR_ST_B>" in type(p.first).__name__ )
      self.assertTrue( "vector<PR_NS_A::PR_NS_D::PR_ST_E>" in type(p.second).__name__ )

   def test04NamespacedTemplateIdentity( self ):
      """Identity of templated classes with and w/o std:: should match"""

      gInterpreter.Declare( 'namespace PR_HepMC { class GenParticle {}; }' )
      gInterpreter.Declare( 'namespace PR_LoKi { template< typename T, typename S > class Functor {}; }' )

      PR_LoKi = ROOT.PR_LoKi

      f1 = PR_LoKi.Functor(      "vector<const PR_HepMC::GenParticle*>",      "vector<double>" )
      f2 = PR_LoKi.Functor( "std::vector<const PR_HepMC::GenParticle*>", "std::vector<double>" )

      self.assertTrue( f1 is f2 )
      self.assertEqual( f1, f2 )



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

    def test1SingleArgumentFunction(self):
        """Test reproducer of issue root-project/root/8817."""
        self.assertEqual(ROOT.Cpp4.myfunc(ROOT.Cpp4.B()), 1)
        self.assertEqual(ROOT.Cpp4.myfunc(ROOT.Cpp4.C()), 2)

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
