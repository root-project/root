# File: roottest/python/basic/PyROOT_datatypetests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 05/11/05
# Last: 01/12/12

"""Data type conversion unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from array import array
from ROOT import *
from common import *

__all__ = [
   'DataTypes1InstanceDataTestCase',
   'DataTypes2ClassDataTestCase',
   'DataTypes3BufferDataTestCase'
]

gROOT.LoadMacro( "DataTypes.C+" )


### access to instance data members ==========================================
class DataTypes1InstanceDataTestCase( MyTestCase ):
 # N = 5 (imported from ROOT as global)

   def test1ReadAccess( self ):
      """Test read access to instance public data and verify values"""

      c = ClassWithData()
      self.failUnless( isinstance( c, ClassWithData ) )

      self.assertEqual( c.fBool, False )
      self.assertEqual( c.fChar,   'a' )
      self.assertEqual( c.fUChar,  'c' )
      self.assertEqual( c.fShort,  -11 )
      self.assertEqual( c.fUShort,  11 )
      self.assertEqual( c.fInt,    -22 )
      self.assertEqual( c.fUInt,    22 )
      self.assertEqual( c.fLong,   pylong(-33) )
      self.assertEqual( c.fULong,  pylong( 33) )
      self.assertEqual( round( c.fFloat  + 44., 5 ), 0 )
      self.assertEqual( round( c.fDouble + 55., 8 ), 0 )

      for i in range(N):
         self.assertEqual( c.fBoolArray[i],         i%2 )
         self.assertEqual( c.GetBoolArray()[i],     i%2 )
         self.assertEqual( c.fBoolArray2[i],     (i+1)%2)
         self.assertEqual( c.GetBoolArray2()[i], (i+1)%2)
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
      self.failUnless( isinstance( c, ClassWithData ) )

    # boolean types
      c.fBool = True;     self.assertEqual( c.GetBool(),  True )
      c.SetBool( True );  self.assertEqual( c.fBool,      True )
      c.fBool = kTRUE;    self.assertEqual( c.GetBool(), kTRUE )
      c.SetBool( kTRUE ); self.assertEqual( c.fBool,     kTRUE )
      self.failUnlessRaises( TypeError, c.SetBool, 10 )

    # char types
      c.fChar = 'b';      self.assertEqual( c.GetChar(),      'b' )
      c.fChar = 40;       self.assertEqual( c.GetChar(),  chr(40) )
      c.SetChar( 'c' );   self.assertEqual( c.fChar,          'c' )
      c.SetChar( 41 );    self.assertEqual( c.fChar,      chr(41) )
      c.fUChar = 'd';     self.assertEqual( c.GetUChar(),     'd' )
      c.fUChar = 42;      self.assertEqual( c.GetUChar(), chr(42) )
      c.SetUChar( 'e' );  self.assertEqual( c.fUChar,         'e' )
      c.SetUChar( 43 );   self.assertEqual( c.fUChar,     chr(43) )

      self.failUnlessRaises( TypeError, c.SetChar,  "string" )
      self.failUnlessRaises( TypeError, c.SetUChar,       -1 )
      self.failUnlessRaises( TypeError, c.SetUChar, "string" )

    # integer types
      names = [ 'Short', 'UShort', 'Int', 'UInt', 'Long', 'ULong' ]
      for i in range(len(names)):
         setattr( c, 'f'+names[i], i )
         self.assertEqual( getattr( c, 'Get'+names[i] )(), i )

      for i in range(len(names)):
         getattr( c, 'Set'+names[i] )( 2*i )
         self.assertEqual( getattr( c, 'f'+names[i] ), 2*i )

    # float types
      c.fFloat = 0.123;     self.assertEqual( round( c.GetFloat()  - 0.123, 5 ), 0 )
      if not FIXCLING:
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
         setattr( c, 'f'+names[j]+'Array', b )    # buffer copies
         for i in range(N):
            self.assertEqual( getattr( c, 'f'+names[j]+'Array' )[i], b[i] )

         setattr( c, 'f'+names[j]+'Array2', b )   # pointer copies
         b[i] = 28
         for i in range(N):
            self.assertEqual( getattr( c, 'f'+names[j]+'Array2' )[i], b[i] )

   def test3RangeAccess( self ):
      """Test the ranges of integer types"""

      def call( c, name, value ):
         setattr( c, name, value )

      c = ClassWithData()
      self.failUnless( isinstance( c, ClassWithData ) )

      self.assertRaises( ValueError, call, c, 'fUInt',  -1  )
      self.assertRaises( ValueError, call, c, 'fULong', -1  )

   def test4TypeConversions( self ):
      """Test conversions between builtin types"""

      c = ClassWithData()

      c.fDouble = -1
      self.assertEqual( c.fDouble, -1.0 )

      self.assertRaises( TypeError, c.fDouble,  'c'  )
      self.assertRaises( TypeError, c.fInt,     -1.  )
      self.assertRaises( TypeError, c.fInt,      1.  )


### access to class data members =============================================
class DataTypes2ClassDataTestCase( MyTestCase ):
   def test1ReadAccess( self ):
      """Test read access to class public data and verify values"""

      c = ClassWithData()

      self.assertEqual( ClassWithData.sChar,    's' )
      self.assertEqual( c.sChar,                's' )
      self.assertEqual( c.sUChar,               'u' )
      self.assertEqual( ClassWithData.sUChar,   'u' )
      self.assertEqual( ClassWithData.sShort,  -101 )
      self.assertEqual( c.sShort,              -101 )
      self.assertEqual( c.sUShort,              255 )
      self.assertEqual( ClassWithData.sUShort,  255 )
      self.assertEqual( ClassWithData.sInt,    -202 )
      self.assertEqual( c.sInt,                -202 )
      self.assertEqual( c.sUInt,                202 )
      self.assertEqual( ClassWithData.sUInt,    202 )
      self.assertEqual( ClassWithData.sLong,   pylong(-303) )
      self.assertEqual( c.sLong,               pylong(-303) )
      self.assertEqual( c.sULong,              pylong( 303) )
      self.assertEqual( ClassWithData.sULong,  pylong( 303) )
      self.assertEqual( round( ClassWithData.sFloat  + 404., 5 ), 0 )
      self.assertEqual( round( c.sFloat              + 404., 5 ), 0 )
      self.assertEqual( round( ClassWithData.sDouble + 505., 8 ), 0 )
      self.assertEqual( round( c.sDouble             + 505., 8 ), 0 )

   def test2WriteAccess( self ):
      """Test write access to class public data and verify values"""

      c = ClassWithData()

      ClassWithData.sChar                    =  'a'
      self.assertEqual( c.sChar,                'a' )
      c.sChar                                =  'b'
      self.assertEqual( ClassWithData.sChar,    'b' )
      ClassWithData.sUChar                   =  'c'
      self.assertEqual( c.sUChar,               'c' )
      c.sUChar                               =  'd'
      self.assertEqual( ClassWithData.sUChar,   'd' )
      self.assertRaises( ValueError, setattr, ClassWithData, 'sUChar', -1 )
      self.assertRaises( ValueError, setattr, c,             'sUChar', -1 )
      c.sShort                               = -102
      self.assertEqual( ClassWithData.sShort,  -102 )
      ClassWithData.sShort                   = -203
      self.assertEqual( c.sShort,              -203 )
      c.sUShort                              =  127
      self.assertEqual( ClassWithData.sUShort,  127 )
      ClassWithData.sUShort                  =  227
      self.assertEqual( c.sUShort,              227 )
      ClassWithData.sInt                     = -234
      self.assertEqual( c.sInt,                -234 )
      c.sInt                                 = -321
      self.assertEqual( ClassWithData.sInt,    -321 )
      ClassWithData.sUInt                    = 1234
      self.assertEqual( c.sUInt,               1234 )
      c.sUInt                                = 4321
      self.assertEqual( ClassWithData.sUInt,   4321 )
      self.assertRaises( ValueError, setattr, c,             'sUInt', -1 )
      self.assertRaises( ValueError, setattr, ClassWithData, 'sUInt', -1 )
      ClassWithData.sLong                    = pylong(-87)
      self.assertEqual( c.sLong,               pylong(-87) )
      c.sLong                                = pylong( 876)
      self.assertEqual( ClassWithData.sLong,   pylong( 876) )
      ClassWithData.sULong                   = pylong( 876)
      self.assertEqual( c.sULong,              pylong( 876) )
      c.sULong                               = pylong( 678)
      self.assertEqual( ClassWithData.sULong,  pylong( 678) )
      self.assertRaises( ValueError, setattr, ClassWithData, 'sULong', -1 )
      self.assertRaises( ValueError, setattr, c,             'sULong', -1 )
      ClassWithData.sFloat                   = -3.1415
      self.assertEqual( round( c.sFloat, 5 ),  -3.1415 )
      c.sFloat                               =  3.1415
      self.assertEqual( round( ClassWithData.sFloat, 5 ), 3.1415 )
      import math
      c.sDouble                              = -math.pi
      self.assertEqual( ClassWithData.sDouble, -math.pi )
      ClassWithData.sDouble                  =  math.pi
      self.assertEqual( c.sDouble,              math.pi )


### access to data through buffer interface ==================================
class DataTypes3BufferDataTestCase( MyTestCase ):
   def test1SetBufferSize( self ):
      """Test usage of buffer sizing"""

      c = ClassWithData()

      for func in [ 'GetShortArray',  'GetShortArray2',
                    'GetUShortArray', 'GetUShortArray2',
                    'GetIntArray',    'GetIntArray2',
                    'GetUIntArray',   'GetUIntArray2',
                    'GetLongArray',   'GetLongArray2',
                    'GetULongArray',  'GetULongArray2' ]:
         arr = getattr( c, func )()
         arr.SetSize( N )
         self.assertEqual( len(arr), N )

         l = list( arr )
         for i in range(N):
            self.assertEqual( arr[i], l[i] )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
