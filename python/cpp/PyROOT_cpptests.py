# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/03/05
# Last: 09/30/10

"""C++ language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ROOT
from ROOT import TObject, TLorentzVector, kRed, kGreen, kBlue, TVectorF, TROOT, TCanvas, gInterpreter, gROOT, TMatrixD, TString, std
from ROOT import MakeNullPointer, AsCObject, BindObject, AddressOf, addressof
from common import *
from functools import partial

__all__ = [
   'Cpp1LanguageFeatureTestCase',
   'Cpp2ClassNamingTestCase'
]



### C++ language constructs test cases =======================================
class Cpp1LanguageFeatureTestCase( MyTestCase ):
   @classmethod
   def setUpClass(cls):
      cls.exp_pyroot = os.environ.get('EXP_PYROOT') == 'True'

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

      if self.exp_pyroot:
         cppname = ROOT.foo.__cpp_name__
      else:
         cppname = ROOT.foo.__cppname__
      self.assertEqual(cppname, 'foo')

      self.assertEqual(ROOT.foo.aa, 0)
      self.assertEqual(ROOT.foo.bb, 1)

   def test04NsEnumType(self):
      """Test lookup type of enum in namespace"""
      ROOT.gInterpreter.Declare("namespace myns { enum foo { aa,bb }; }")

      self.assertEqual(ROOT.myns.aa, 0)
      self.assertEqual(ROOT.myns.bb, 1)

      if self.exp_pyroot:
         cppname = ROOT.myns.foo.__cpp_name__
      else:
         cppname = ROOT.myns.foo.__cppname__
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

      if self.exp_pyroot:
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
      else:
         t4 = TLorentzVector(0, 0, 0, 0)
         t4.__init__(TLorentzVector(0, 1, 2, 3))
         # the following should work exactly as above, but no longer does on some version of ROOT 6
         t5 = TLorentzVector(0, 0, 0, 0)
         TLorentzVector.__init__(t5, TLorentzVector(0, 1, 2, 3))

         for i in range(4):
            self.assertEqual( t4[i], t5[i] )

   def test08ObjectValidity( self ):
      """Test object validity checking"""

      t1 = TObject()

      self.assert_( t1 )
      self.assert_( not not t1 )

      t2 = gROOT.FindObject( "Nah, I don't exist" )

      self.assert_( not t2 )

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
      self.assert_( not not c1 )

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
      if hasattr( array.array, 'buffer_info' ):   # not supported in p2.2
         if self.exp_pyroot:
            # New cppyy uses unsigned long to represent void* returns, as in DynamicCast.
            # To prevent an overflow error when converting the Python integer returned by
            # DynamicCast into a 4-byte signed long in 32 bits, we use unsigned long ('L')
            # as type of the array.array.
            array_t = 'L'
         else:
            # Old PyROOT returns Long_t buffers for void*
            array_t = 'l'
         addressofo = array.array( array_t, [o.IsA()._TClass__DynamicCast( o.IsA(), o )[0]] )
         self.assertEqual( addressofo.buffer_info()[0], Z.GimeAddressPtrPtr( addressofo ) )

      self.assertEqual( 0, Z.GimeAddressPtr( 0 ) );
      self.assertEqual( 0, Z.GimeAddressObject( 0 ) );
      if not self.exp_pyroot:
         # The conversion None -> ptr is not supported in new Cppyy
         self.assertEqual( 0, Z.GimeAddressPtr( None ) );
         self.assertEqual( 0, Z.GimeAddressObject( None ) );

      ptr = MakeNullPointer( TObject )
      if self.exp_pyroot:
         # New Cppyy does not raise ValueError,
         # it just returns zero
         self.assertEqual(addressof(ptr), 0)
      else:
         self.assertRaises( ValueError, AddressOf, ptr )
      Z.SetAddressPtrRef( ptr )

      self.assertEqual( addressof( ptr ), 0x1234 )
      Z.SetAddressPtrPtr( ptr )
      self.assertEqual( addressof( ptr ), 0x4321 )

   def test13Macro( self ):
      """Test access to cpp macro's"""
      if not self.exp_pyroot:
         # In new PyROOT, we will just provide ROOT.nullptr
         self.assertEqual( ROOT.NULL, 0 );

      gROOT.ProcessLine( '#define aap "aap"' )
      gROOT.ProcessLine( '#define noot 1' )
      gROOT.ProcessLine( '#define mies 2.0' )

      # looking up macro's is slow, so needs to be explicit (note that NULL,
      # see above, is a special case)
      ROOT.PyConfig.ExposeCppMacros = True

      if not self.exp_pyroot:
         # New Cppyy does not do lookup of macros
         self.assertEqual( ROOT.aap, "aap" )
         self.assertEqual( ROOT.noot, 1 )
         self.assertEqual( ROOT.mies, 2.0 )

      # test also that garbage macros are not found
      self.assertRaises( AttributeError, getattr, ROOT, "_this_does_not_exist_at_all" )

      ROOT.PyConfig.ExposeCppMacros = False

   def test14OpaquePointerPassing( self ):
      """Test passing around of opaque pointers"""

      import ROOT

      s = TString( "Hello World!" )
      co = AsCObject( s )
      
      ad = addressof( s )

      self.assert_( s == BindObject( co, s.__class__ ) )
      self.assert_( s == BindObject( co, "TString" ) )
      self.assert_( s == BindObject( ad, s.__class__ ) )
      self.assert_( s == BindObject( ad, "TString" ) )

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

      self.assert_( l3 != None )        # like this to ensure __ne__ is called
      self.assert_( None != l3 )        # id.
      self.assertNotEqual( l3, l5 )
      self.assertNotEqual( l5, l3 )

   def test16AddressOfaddressof(self):
      """Test addresses returned by AddressOf and addressof"""
      import ROOT

      o = ROOT.TObject()

      addr_as_int    = ROOT.addressof(o)
      addr_as_buffer = ROOT.AddressOf(o)

      # The result of AddressOf can be passed to a function that expects a void*
      ROOT.gInterpreter.Declare("""
      long long get_address_in_buffer(void *p) { return *(long long*)p; }
      """)

      self.assertEqual(addr_as_int, ROOT.get_address_in_buffer(addr_as_buffer))


### C++ language naming of classes ===========================================
class Cpp2ClassNamingTestCase( MyTestCase ):
   def test01Underscore( self ):
      """Test recognition of '_' as part of a valid class name"""

      z = ROOT.Z_()

      self.assert_( hasattr( z, 'myint' ) )
      self.assert_( z.GimeZ_( z ) )

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
      self.assert_( "vector<PR_NS_A::PR_ST_B>" in type(p.first).__name__ )
      self.assert_( "vector<PR_NS_A::PR_NS_D::PR_ST_E>" in type(p.second).__name__ )

   def test04NamespacedTemplateIdentity( self ):
      """Identity of templated classes with and w/o std:: should match"""

      gInterpreter.Declare( 'namespace PR_HepMC { class GenParticle {}; }' )
      gInterpreter.Declare( 'namespace PR_LoKi { template< typename T, typename S > class Functor {}; }' )

      PR_LoKi = ROOT.PR_LoKi

      f1 = PR_LoKi.Functor(      "vector<const PR_HepMC::GenParticle*>",      "vector<double>" )
      f2 = PR_LoKi.Functor( "std::vector<const PR_HepMC::GenParticle*>", "std::vector<double>" )

      self.assert_( f1 is f2 )
      self.assertEqual( f1, f2 )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
