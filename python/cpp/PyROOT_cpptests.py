# File: roottest/python/cpp/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/03/05
# Last: 03/18/05

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

      self.assertEqual( TObject.kBitMask,    0xffffff )
      self.assertEqual( TObject.kIsOnHeap,   0x1000000 )
      self.assertEqual( TObject.kNotDeleted, 0x2000000 )
      self.assertEqual( TObject.kZombie,     0x4000000 )

      t = TObject()

      self.assertEqual( TObject.kBitMask,    t.kBitMask )
      self.assertEqual( TObject.kIsOnHeap,   t.kIsOnHeap )
      self.assertEqual( TObject.kNotDeleted, t.kNotDeleted )
      self.assertEqual( TObject.kZombie,     t.kZombie )

   def test2Globalenum( self ):
      """Test global enums access and values"""

      self.assertEqual( kRed,   0x2 )
      self.assertEqual( kGreen, 0x3 )
      self.assertEqual( kBlue,  0x4 )

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


## actual test run
if __name__ == '__main__':
   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = unittest.TextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
