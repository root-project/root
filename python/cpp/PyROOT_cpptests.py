# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/03/05
# Last: 03/09/07

"""C++ language interface unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Cpp1LanguageFeatureTestCase'
]


### C++ language constructs test cases =======================================
class Cpp1LanguageFeatureTestCase( unittest.TestCase ):
   def test1ClassEnum( self ):
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

   def test2Globalenum( self ):
      """Test global enums access and values"""

      self.assertEqual( kRed,   gROOT.ProcessLine( "return kRed;" ) )
      self.assertEqual( kGreen, gROOT.ProcessLine( "return kGreen;" ) )
      self.assertEqual( kBlue,  gROOT.ProcessLine( "return kBlue;" ) )

   def test3CopyContructor( self ):
      """Test copy constructor"""

      t1 = TLorentzVector( 1., 2., 3., -4. )
      t2 = TLorentzVector( 0., 0., 0.,  0. )
      t3 = TLorentzVector( t1 )

      self.assertEqual( t1, t3 )
      self.assertNotEqual( t1, t2 )

      for i in range(4):
         self.assertEqual( t1[i], t3[i] )

   def test4ObjectValidity( self ):
      """Test object validity checking"""

      t1 = TObject()

      self.assert_( t1 )
      self.assert_( not not t1 )

      t2 = gROOT.FindObject( "Nah, I don't exist" )

      self.assert_( not t2 )

   def test5ElementAccess( self ):
      """Test access to elements in matrix and array objects."""

      n = 3
      v = TVectorF( n )
      m = TMatrixD( n, n )

      for i in range(n):
         self.assertEqual( v[i], 0.0 )

         for j in range(n):
             self.assertEqual( m[i][j], 0.0 )

   def test6StaticFunctionCall( self ):
      """Test call to static function."""

      c1 = TString.Class()
      self.assert_( not not c1 )

      s = TString()
      c2 = s.Class()

      self.assertEqual( c1, c2 )

      old = s.InitialCapacity( 20 )
      self.assertEqual( 20, TString.InitialCapacity( old ) )

      old = TString.InitialCapacity( 20 )
      self.assertEqual( 20, s.InitialCapacity( old ) )

   def test7Namespaces( self ):
      """Test access to namespaces and inner classes"""

      gROOT.LoadMacro( "Namespace.C+" )

      self.assertEqual( A.sa,          1 )
      self.assertEqual( A.B.sb,        2 )
      self.assertEqual( A.B().fb,     -2 )
      self.assertEqual( A.B.C.sc,      3 )
      self.assertEqual( A.B.C().fc,   -3 )
      self.assertEqual( A.D.sd,        4 )
      self.assertEqual( A.D.E.se,      5 )
      self.assertEqual( A.D.E().fe,   -5 )
      self.assertEqual( A.D.E.F.sf,    6 )
      self.assertEqual( A.D.E.F().ff, -6 )

   def test8VoidPointerPassing( self ):
      """Test passing of variants of void pointer arguments"""

      gROOT.LoadMacro( "PointerPassing.C+" )

      o = TObject()
      self.assertEqual( AddressOf( o )[0], Z.GimeAddressPtr( o ) )
      self.assertEqual( AddressOf( o )[0], Z.GimeAddressPtrRef( o ) )

      import array
      if hasattr( array.array, 'buffer_info' ):   # not supported in p2.2
         addressofo = array.array( 'l', [o.IsA()._TClass__DynamicCast( o.IsA(), o )] )
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

   def test9Macro( self ):
      """Test access to cpp macro's"""

      self.assertEqual( NULL, 0 );

      gROOT.ProcessLine( '#define aap "aap"' )
      gROOT.ProcessLine( '#define noot 1' )
      gROOT.ProcessLine( '#define mies 2.0' )

      self.assertEqual( aap, "aap" )
      self.assertEqual( noot, 1 )
      self.assertEqual( mies, 2.0 )

## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
