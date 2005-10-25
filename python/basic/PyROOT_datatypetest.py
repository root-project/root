# File: roottest/python/basic/PyROOT_datatypetests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 05/11/05
# Last: 05/26/05

"""Data type conversion unit tests for PyROOT package."""

import os, sys, unittest
from array import array
from ROOT import *

__all__ = [
   'DataTypes1InstanceDataTestCase'
]

gROOT.LoadMacro( "DataTypes.C+" )


### access to instance data members ==========================================
class DataTypes1InstanceDataTestCase( unittest.TestCase ):
 # N = 5 (imported from ROOT as global)

   def test1ReadAccess( self ):
      """Test read access to instance public data and verify values"""

      c = ClassWithData()

      self.assertEqual( c.fChar,   'a' )
      self.assertEqual( c.fUChar,  'c' )
      self.assertEqual( c.fShort,  -11 )
      self.assertEqual( c.fUShort,  11 )
      self.assertEqual( c.fInt,    -22 )
      self.assertEqual( c.fUInt,    22 )
      self.assertEqual( c.fLong,   -33L )
      self.assertEqual( c.fULong,   33L )
      self.assertEqual( round( c.fFloat  + 44., 5 ), 0 )
      self.assertEqual( round( c.fDouble + 55., 8 ), 0 )

      for i in range(N):
         self.assertEqual( c.fShortArray[i],       -1*i )
         self.assertEqual( c.GetShortArray()[i],   -1*i )
         self.assertEqual( c.fShortArray2[i],      -2*i )
         self.assertEqual( c.GetShortArray2()[i],  -2*i )
         self.assertEqual( c.fUShortArray[i],       3*i )
         self.assertEqual( c.GetUShortArray()[i],   3*i )
         self.assertEqual( c.fUShortArray2[i],      4*i )
         self.assertEqual( c.GetUShortArray2()[i],  4*i )
         self.assertEqual( c.fIntArray[i],         -5*i )
         self.assertEqual( c.GetIntArray()[i],     -5*i )
         self.assertEqual( c.fIntArray2[i],        -6*i )
         self.assertEqual( c.GetIntArray2()[i],    -6*i )
         self.assertEqual( c.fUIntArray[i],         7*i )
         self.assertEqual( c.GetUIntArray()[i],     7*i )
         self.assertEqual( c.fUIntArray2[i],        8*i )
         self.assertEqual( c.GetUIntArray2()[i],    8*i )
         self.assertEqual( c.fLongArray[i],        -9*i )
         self.assertEqual( c.GetLongArray()[i],    -9*i )
         self.assertEqual( c.fLongArray2[i],      -10*i )
         self.assertEqual( c.GetLongArray2()[i],  -10*i )
         self.assertEqual( c.fULongArray[i],       11*i )
         self.assertEqual( c.GetULongArray()[i],   11*i )
         self.assertEqual( c.fULongArray2[i],      12*i )
         self.assertEqual( c.GetULongArray2()[i],  12*i )

         self.assertEqual( round( c.fFloatArray[i]   + 13.*i, 5 ), 0 )
         self.assertEqual( round( c.fFloatArray2[i]  + 14.*i, 5 ), 0 )
         self.assertEqual( round( c.fDoubleArray[i]  + 15.*i, 8 ), 0 )
         self.assertEqual( round( c.fDoubleArray2[i] + 16.*i, 8 ), 0 )

      self.assertRaises( IndexError, c.fShortArray.__getitem__,      N )
      self.assertRaises( IndexError, c.fUShortArray.__getitem__,     N )
      self.assertRaises( IndexError, c.fIntArray.__getitem__,        N )
      self.assertRaises( IndexError, c.fUIntArray.__getitem__,       N )
      self.assertRaises( IndexError, c.fLongArray.__getitem__,       N )
      self.assertRaises( IndexError, c.fULongArray.__getitem__,      N )
      self.assertRaises( IndexError, c.fFloatArray.__getitem__,      N )
      self.assertRaises( IndexError, c.fDoubleArray.__getitem__,     N )

   def test2WriteAccess( self ):
      """Test write access to instance public data and verify values"""

      c = ClassWithData()

    # char types
      c.fChar = 'b';     self.assertEqual( c.GetChar(),  'b' )
      c.SetChar( 'c' );  self.assertEqual( c.fChar,      'c' )
      c.fUChar = 'd';    self.assertEqual( c.GetUChar(), 'd' )
      c.SetUChar( 'e' ); self.assertEqual( c.fUChar,     'e' )

    # integer types
      names = [ 'Short', 'UShort', 'Int', 'UInt', 'Long', 'ULong' ]
      for i in range(len(names)):
         exec 'c.f%s = %d' % (names[i],i)
         self.assertEqual( eval( 'c.Get%s()' % names[i] ), i )

      for i in range(len(names)):
         exec 'c.Set%s = %d' % (names[i],2*i)
         self.assertEqual( eval( 'c.f%s' % names[i] ), i )

    # float types
      c.fFloat = 0.123;     self.assertEqual( round( c.GetFloat()  - 0.123, 5 ), 0 )
      c.SetFloat( 0.234 );  self.assertEqual( round( c.fFloat      - 0.234, 5 ), 0 )
      c.fDouble = 0.456;    self.assertEqual( round( c.GetDouble() - 0.456, 8 ), 0 )
      c.SetDouble( 0.567 ); self.assertEqual( round( c.fDouble     - 0.567, 8 ), 0 )

    # arrays; there will be pointer copies, so destroy the current ones
      c.DestroyArrays()

    # integer arrays
      a = range(N)
      atypes = [ 'h', 'H', 'i', 'I', 'l', 'L' ]
      for j in range(len(names)):
         b = array( atypes[j], a )
         exec 'c.f%sArray = b' % names[j]   # buffer copies
         for i in range(N):
            exec 'self.assertEqual( c.f%sArray[i], b[i] )' % names[j]

         exec 'c.f%sArray2 = b' % names[j]  # pointer copies
         b[i] = 28
         for i in range(N):
            exec 'self.assertEqual( c.f%sArray2[i], b[i] )' % names[j]


### access to class data members =============================================
class DataTypes2ClassDataTestCase( unittest.TestCase ):
   def test1ReadAccess( self ):
      """Test read access to class public data and verify values"""

      self.assertEqual( ClassWithData.sChar,    's' )
      self.assertEqual( ClassWithData.sUChar,   'u' )
      self.assertEqual( ClassWithData.sShort,  -101 )
      self.assertEqual( ClassWithData.sUShort,  255 )
      self.assertEqual( ClassWithData.sInt,    -202 )
      self.assertEqual( ClassWithData.sUInt,    202 )
      self.assertEqual( ClassWithData.sLong,   -303L )
      self.assertEqual( ClassWithData.sULong,   303L )
      self.assertEqual( round( ClassWithData.sFloat  + 404., 5 ), 0 )
      self.assertEqual( round( ClassWithData.sDouble + 505., 8 ), 0 )


### access to global data members ============================================


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
