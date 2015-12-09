# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/03/05
# Last: 09/30/10

"""C++ language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import *

__all__ = [
   'Cpp1LanguageFeatureTestCase',
   'Cpp2ClassNamingTestCase'
]


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

   def test03CopyContructor( self ):
      """Test copy constructor"""

      t1 = TLorentzVector( 1., 2., 3., -4. )
      t2 = TLorentzVector( 0., 0., 0.,  0. )
      t3 = TLorentzVector( t1 )

      self.assertEqual( t1, t3 )
      self.assertNotEqual( t1, t2 )

      for i in range(4):
         self.assertEqual( t1[i], t3[i] )

   def test04ObjectValidity( self ):
      """Test object validity checking"""

      t1 = TObject()

      self.assert_( t1 )
      self.assert_( not not t1 )

      t2 = gROOT.FindObject( "Nah, I don't exist" )

      self.assert_( not t2 )

   def test05ElementAccess( self ):
      """Test access to elements in matrix and array objects."""

      n = 3
      v = TVectorF( n )
      m = TMatrixD( n, n )

      for i in range(n):
         self.assertEqual( v[i], 0.0 )

         for j in range(n):
            self.assertEqual( m[i][j], 0.0 )

   def test06StaticFunctionCall( self ):
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

   def test07Namespaces( self ):
      """Test access to namespaces and inner classes"""

      gROOT.LoadMacro( "Namespace.C+" )

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

   def test08VoidPointerPassing( self ):
      """Test passing of variants of void pointer arguments"""

      gROOT.LoadMacro( "PointerPassing.C+" )

      o = TObject()
      self.assertEqual( AddressOf( o )[0], Z.GimeAddressPtr( o ) )
      self.assertEqual( AddressOf( o )[0], Z.GimeAddressPtrRef( o ) )
      
      pZ = Z.getZ(0)
      self.assertEqual( Z.checkAddressOfZ( pZ ), True )
      self.assertEqual( pZ , Z.getZ(1) )

      import array
      if hasattr( array.array, 'buffer_info' ):   # not supported in p2.2
         addressofo = array.array( 'l', [o.IsA()._TClass__DynamicCast( o.IsA(), o )[0]] )
         self.assertEqual( addressofo.buffer_info()[0], Z.GimeAddressPtrPtr( addressofo ) )

      self.assertEqual( 0, Z.GimeAddressPtr( 0 ) );
      self.assertEqual( 0, Z.GimeAddressPtr( None ) );
      self.assertEqual( 0, Z.GimeAddressObject( 0 ) );
      self.assertEqual( 0, Z.GimeAddressObject( None ) );

      ptr = MakeNullPointer( TObject )
      self.assertRaises( ValueError, AddressOf, ptr )
      Z.SetAddressPtrRef( ptr )
      self.assertEqual( AddressOf( ptr )[0], 0x1234 )
      Z.SetAddressPtrPtr( ptr )
      self.assertEqual( AddressOf( ptr )[0], 0x4321 )

   def test09Macro( self ):
      """Test access to cpp macro's"""

      self.assertEqual( NULL, 0 );

      gROOT.ProcessLine( '#define aap "aap"' )
      gROOT.ProcessLine( '#define noot 1' )
      gROOT.ProcessLine( '#define mies 2.0' )

      # looking up macro's is slow, so needs to be explicit (note that NULL,
      # see above, is a special case)
      import ROOT
      ROOT.PyConfig.ExposeCppMacros = True

      self.assertEqual( ROOT.aap, "aap" )
      self.assertEqual( ROOT.noot, 1 )
      self.assertEqual( ROOT.mies, 2.0 )

      # test also that garbage macros are not found
      self.assertRaises( AttributeError, getattr, ROOT, "_this_does_not_exist_at_all" )

      ROOT.PyConfig.ExposeCppMacros = False

   def test10OpaquePointerPassing( self ):
      """Test passing around of opaque pointers"""

      import ROOT

      s = TString( "Hello World!" )
      co = ROOT.AsCObject( s )
      ad = ROOT.AddressOf( s )[ 0 ]

      self.assert_( s == ROOT.BindObject( co, s.__class__ ) )
      self.assert_( s == ROOT.BindObject( co, "TString" ) )
      self.assert_( s == ROOT.BindObject( ad, s.__class__ ) )
      self.assert_( s == ROOT.BindObject( ad, "TString" ) )

   def test11ObjectAndPointerComparisons( self ):
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


### C++ language naming of classes ===========================================
class Cpp2ClassNamingTestCase( MyTestCase ):
   def test01Underscore( self ):
      """Test recognition of '_' as part of a valid class name"""

      z = Z_()

      self.assert_( hasattr( z, 'myint' ) )
      self.assert_( z.GimeZ_( z ) )

   def test02DefaultCtorInNamespace( self ):
      """Check that constructor with default argument is found in namespace"""

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

      p = std.pair( std.vector( PR_NS_A.PR_ST_B ), std.vector( PR_NS_A.PR_NS_D.PR_ST_E ) )()
      self.assert_( "vector<PR_NS_A::PR_ST_B>" in type(p.first).__name__ )
      self.assert_( "vector<PR_NS_A::PR_NS_D::PR_ST_E>" in type(p.second).__name__ )

   def test04NamespacedTemplateIdentity( self ):
      """Identity of templated classes with and w/o std:: should match"""

      gInterpreter.Declare( 'namespace PR_HepMC { class GenParticle {}; }' )
      gInterpreter.Declare( 'namespace PR_LoKi { template< typename T, typename S > class Functor {}; }' )

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
