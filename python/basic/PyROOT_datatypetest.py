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
   'DataTypes3BufferDataTestCase',
   'DataTypes4RegressionTestCase'
]

gROOT.LoadMacro( "DataTypes.C+" )


### access to instance data members ==========================================
class DataTypes1InstanceDataTestCase( MyTestCase ):
 # N = 5 (imported from ROOT as global)

   def test1ReadAccess( self ):
      """Test read access to instance public data and verify values"""

      c = ClassWithData()
      self.failUnless( isinstance( c, ClassWithData ) )

      self.assertEqual( c.fBool,  False )
      self.assertEqual( c.fChar,    'a' )
      self.assertEqual( c.fSChar,   'b' )
      self.assertEqual( c.fUChar,   'c' )
      self.assertEqual( c.fShort,   -11 )
      self.assertEqual( c.fUShort,   11 )
      self.assertEqual( c.fInt,     -22 )
      self.assertEqual( c.fUInt,     22 )
      self.assertEqual( c.fLong,    pylong(-33) )
      self.assertEqual( c.fULong,   pylong( 33) )
      self.assertEqual( c.fLong64,  pylong(-44) )
      self.assertEqual( c.fULong64, pylong( 44) )
      self.assertEqual( round( c.fFloat  + 55., 5 ), 0 )
      self.assertEqual( round( c.fDouble + 66., 8 ), 0 )
      self.assertEqual( c.fEnum,    ClassWithData.kNothing )
      self.assertEqual( c.fEnum,    c.kNothing )

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
      c.fSChar = 'd';     self.assertEqual( c.GetSChar(),     'd' )
      c.fSChar = 42;      self.assertEqual( c.GetSChar(), chr(42) )
      c.fUChar = 'e';     self.assertEqual( c.GetUChar(),     'e' )
      c.fUChar = 43;      self.assertEqual( c.GetUChar(), chr(43) )
      c.SetUChar( 'f' );  self.assertEqual( c.fUChar,         'f' )
      c.SetUChar( 44 );   self.assertEqual( c.fUChar,     chr(44) )

      self.failUnlessRaises( TypeError, c.SetChar,  "string" )
      self.failUnlessRaises( TypeError, c.SetSChar, "string" )
      self.failUnlessRaises( TypeError, c.SetUChar,       -1 )
      self.failUnlessRaises( TypeError, c.SetUChar, "string" )

    # integer types
      names = [ 'Short', 'UShort', 'Int', 'UInt', 'Long', 'ULong', 'Long64', 'ULong64' ]
      for i in range(len(names)):
         setattr( c, 'f'+names[i], i )
         self.assertEqual( getattr( c, 'Get'+names[i] )(), i )

      for i in range(len(names)):
         getattr( c, 'Set'+names[i] )( 2*i )
         self.assertEqual( getattr( c, 'f'+names[i] ), 2*i )

    # float types
      c.fFloat = 0.123;     self.assertEqual( round( c.GetFloat()  - 0.123, 5 ), 0 )
      c.SetFloat( 0.234 );  self.assertEqual( round( c.fFloat      - 0.234, 5 ), 0 )
      c.fDouble = 0.456;    self.assertEqual( round( c.GetDouble() - 0.456, 8 ), 0 )
      c.SetDouble( 0.567 ); self.assertEqual( round( c.fDouble     - 0.567, 8 ), 0 )

    # enum types
      c.fEnum = ClassWithData.kSomething; self.assertEqual( c.GetEnum(), c.kSomething )
      c.SetEnum( ClassWithData.kLots );   self.assertEqual( c.fEnum, c.kLots )

    # arrays; there will be pointer copies, so destroy the current ones
      c.DestroyArrays()

    # integer arrays
      a = range(N)
      atypes = [ 'h', 'H', 'i', 'I', 'l', 'L' ]
      for j in range(len(names) - 2):   # skip Long64 and ULong64
         b = array( atypes[j], a )
         setattr( c, 'f'+names[j]+'Array', b )    # buffer copies
         for i in range(N):
            self.assertEqual( getattr( c, 'f'+names[j]+'Array' )[i], b[i] )

         setattr( c, 'f'+names[j]+'Array2', b )   # pointer copies
         b[i] = 28
         for i in range(N):
            self.assertEqual( getattr( c, 'f'+names[j]+'Array2' )[i], b[i] )

   def test3WriteAccessThroughConstRef( self ):
      """Const ref is by-ptr for Cling, verify independently"""

      c = ClassWithData()
      self.failUnless( isinstance( c, ClassWithData ) )

    # boolean types
      c.SetBoolCR( True );  self.assertEqual( c.fBool,      True )
      c.SetBoolCR( kTRUE ); self.assertEqual( c.fBool,     kTRUE )
      self.failUnlessRaises( TypeError, c.SetBoolCR, 10 )

    # char types
      c.SetCharCR( 'c' );   self.assertEqual( c.fChar,       'c' )
      c.SetUCharCR( 'f' );  self.assertEqual( c.fUChar,      'f' )

    # integer types
      names = [ 'Short', 'UShort', 'Int', 'UInt', 'Long', 'ULong', 'Long64', 'ULong64' ]
      for i in range(len(names)):
         getattr( c, 'Set'+names[i]+'CR' )( 2*i )
         self.assertEqual( getattr( c, 'f'+names[i] ), 2*i )

    # float types
      c.SetFloatCR( 0.234 );  self.assertEqual( round( c.fFloat      - 0.234, 5 ), 0 )
      c.SetDoubleCR( 0.567 ); self.assertEqual( round( c.fDouble     - 0.567, 8 ), 0 )

    # enum types
      c.SetEnumCR( ClassWithData.kLots );   self.assertEqual( c.fEnum, c.kLots )

    # arrays; there will be pointer copies, so destroy the current ones
      c.DestroyArrays()

   def test4RangeAccess( self ):
      """Test the ranges of integer types"""

      def call( c, name, value ):
         setattr( c, name, value )

      c = ClassWithData()
      self.failUnless( isinstance( c, ClassWithData ) )

      self.assertRaises( ValueError, call, c, 'fUInt',    -1 )
      self.assertRaises( ValueError, call, c, 'fULong',   -1 )
      self.assertRaises( ValueError, call, c, 'fULong64', -1 )

   def test5TypeConversions( self ):
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

      self.assertEqual( c.sBool,               False )
      self.assertEqual( ClassWithData.sBool,   False )
      self.assertEqual( ClassWithData.sChar,     's' )
      self.assertEqual( c.sChar,                 's' )
      self.assertEqual( c.sSChar,                'S' )
      self.assertEqual( ClassWithData.sSChar,    'S' )
      self.assertEqual( c.sUChar,                'u' )
      self.assertEqual( ClassWithData.sUChar,    'u' )
      self.assertEqual( ClassWithData.sShort,   -101 )
      self.assertEqual( c.sShort,               -101 )
      self.assertEqual( c.sUShort,               255 )
      self.assertEqual( ClassWithData.sUShort,   255 )
      self.assertEqual( ClassWithData.sInt,     -202 )
      self.assertEqual( c.sInt,                 -202 )
      self.assertEqual( c.sUInt,                 202 )
      self.assertEqual( ClassWithData.sUInt,     202 )
      self.assertEqual( ClassWithData.sLong,    pylong(-303) )
      self.assertEqual( c.sLong,                pylong(-303) )
      self.assertEqual( c.sULong,               pylong( 303) )
      self.assertEqual( ClassWithData.sULong,   pylong( 303) )
      self.assertEqual( ClassWithData.sLong64,  pylong(-404) )
      self.assertEqual( c.sLong64,              pylong(-404) )
      self.assertEqual( c.sULong64,             pylong( 404) )
      self.assertEqual( ClassWithData.sULong64, pylong( 404) )
      self.assertEqual( round( ClassWithData.sFloat  + 505., 5 ), 0 )
      self.assertEqual( round( c.sFloat              + 505., 5 ), 0 )
      self.assertEqual( round( ClassWithData.sDouble + 606., 8 ), 0 )
      self.assertEqual( round( c.sDouble             + 606., 8 ), 0 )
      self.assertEqual( ClassWithData.sEnum,    kApple )
      self.assertEqual( c.sEnum,                kApple )

   def test2WriteAccess( self ):
      """Test write access to class public data and verify values"""

      c = ClassWithData()

      ClassWithData.sBool                    = True
      self.assertEqual( c.sBool,               True )
      c.sBool                                = False
      self.assertEqual( ClassWithData.sBool,   False )
      ClassWithData.sChar                    =  'a'
      self.assertEqual( c.sChar,                'a' )
      c.sChar                                =  'b'
      self.assertEqual( ClassWithData.sChar,    'b' )
      ClassWithData.sSChar                   =  'b'
      self.assertEqual( c.sSChar,               'b' )
      c.sSChar                               =  'c'
      self.assertEqual( ClassWithData.sSChar,   'c' )
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
      ClassWithData.sLong64                    = pylong(-90)
      self.assertEqual( c.sLong64,               pylong(-90) )
      c.sLong64                                = pylong( 901)
      self.assertEqual( ClassWithData.sLong64,   pylong( 901) )
      ClassWithData.sULong64                   = pylong( 901)
      self.assertEqual( c.sULong64,              pylong( 901) )
      c.sULong64                               = pylong( 321)
      self.assertEqual( ClassWithData.sULong64,  pylong( 321) )
      self.assertRaises( ValueError, setattr, ClassWithData, 'sULong64', -1 )
      self.assertRaises( ValueError, setattr, c,             'sULong64', -1 )
      ClassWithData.sFloat                   = -3.1415
      self.assertEqual( round( c.sFloat, 5 ),  -3.1415 )
      c.sFloat                               =  3.1415
      self.assertEqual( round( ClassWithData.sFloat, 5 ), 3.1415 )
      import math
      c.sDouble                              = -math.pi
      self.assertEqual( ClassWithData.sDouble, -math.pi )
      ClassWithData.sDouble                  =  math.pi
      self.assertEqual( c.sDouble,              math.pi )
      if not FIXCLING:
         c.sEnum                                = kBanana
         self.assertEqual( ClassWithData.sEnum,   kBanana )
         ClassWithData.sEnum                    = kCitrus
         self.assertEqual( c.sEnum ,              kCitrus )


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


### regression tests after bug reports =======================================
class DataTypes4RegressionTestCase( MyTestCase ):
   def test1ConstBoolRefReturn( self ):
      """Used to fail executor and then crash"""

      c = ClassWithData()

      c.SetBool( False )
      self.assertEqual( c.GetBoolCR(), False )

      c.SetBool( True )
      self.assertEqual( c.GetBoolCR(), True )

   def test2ConstRefReturns( self ):
      """General test of const-ref returns"""

      c = ClassWithData()

      c.SetChar( 'a' );     self.assertEqual( c.GetCharCR(),    'a' )
      c.SetSChar( 'b' );    self.assertEqual( c.GetSCharCR(),   'b' )
      c.SetUChar( 'c' );    self.assertEqual( c.GetUCharCR(),   'c' )
      c.SetShort( -1 );     self.assertEqual( c.GetShortCR(),   -1  )
      c.SetUShort( 1 );     self.assertEqual( c.GetUShortCR(),   1  )
      c.SetInt( -2 );       self.assertEqual( c.GetIntCR(),     -2  )
      c.SetUInt( 2 );       self.assertEqual( c.GetUIntCR(),     2  )
      c.SetLong( -3 );      self.assertEqual( c.GetLongCR(),    -3  )
      c.SetULong( 3 );      self.assertEqual( c.GetULongCR(),    3  )
      c.SetLong64( -4 );    self.assertEqual( c.GetLong64CR(),  -4  )
      c.SetULong64( 4 );    self.assertEqual( c.GetULong64CR(),  4  )
      c.SetFloat( 3.14 );   self.assertEqual( round( c.GetFloatCR()  - 3.14, 5 ), 0 )
      c.SetDouble( 2.72 );  self.assertEqual( round( c.GetDoubleCR() - 2.72, 8 ), 0 )
      c.SetEnum( c.kLots ); self.assertEqual( c.GetEnumCR(), c.kLots )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
