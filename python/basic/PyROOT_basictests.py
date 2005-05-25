# File: roottest/python/basic/PyROOT_basictests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 11/23/04
# Last: 05/11/05

"""Basic unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'Basic1ModuleTestCase',
   'Basic2SetupTestCase',
   'Basic3PythonLanguageTestCase',
   'Basic4PythonizationTestCase'
]


### basic module test cases ==================================================
class Basic1ModuleTestCase( unittest.TestCase ):
   def test1Import( self ):
      """Test import error handling"""

      def failImport():
         from ROOT import GatenKaas

      self.assertRaises( ImportError, failImport )


### basic functioning test cases =============================================
class Basic2SetupTestCase( unittest.TestCase ):
   def test1Globals( self ):
      """Test the availability of ROOT globals"""

      self.assert_( gROOT )
      self.assert_( gApplication )
      self.assert_( gSystem )
      self.assert_( gInterpreter )
      self.assert_( gDirectory )

   def test2AutoLoading( self ):
      """Test auto-loading by retrieving a non-preloaded class"""

      t = TLorentzVector()

   def test3MacroLoading( self ):
      """Test accessibility to macro classes"""

      gROOT.LoadMacro( 'SimpleClass.C' )

      self.assert_( issubclass( SimpleClass, TheBase ) )
      self.assertEqual( SimpleClass, SimpleClass_t )

      c = SimpleClass()
      self.assertEqual( c.fData, c.GetData() )

      c.SetData( 13 )
      self.assertEqual( c.fData, 13 )
      self.assertEqual( c.GetData(), 13 )


### basic python language features test cases ================================
class Basic3PythonLanguageTestCase( unittest.TestCase ):
   def test1HaveDocString( self ):
      """Test doc strings existence"""

      self.assert_( hasattr( TCanvas, "__doc__" ) )
      self.assert_( hasattr( TCanvas.__init__, "__doc__" ) )

   def test2BoundUnboundMethodCalls( self ):
      """Test (un)bound method calls"""

      self.assertRaises( TypeError, TLorentzVector.X )

      m = TLorentzVector.X
      self.assertRaises( TypeError, m )
      self.assertRaises( TypeError, m, 1 )

      b = TLorentzVector()
      b.SetX( 1.0 )
      self.assertEqual( 1.0, m( b ) )


### basic extension features test cases ======================================
class Basic4PythonizationTestCase( unittest.TestCase ):
   def test1Strings( self ):
      """Test string/TString/TObjString compatibility"""

      pyteststr = "aap noot mies"

      s1 = TString( pyteststr )
      s2 = str( s1 )

      self.assertEqual( s1, s2 )

      s3 = TObjString( s2 )
      self.assertEqual( s2, s3 )
      self.assertEqual( s2, pyteststr )

   def test2Lists( self ):
      """Test list/TList behaviour and compatibility"""

      l = TList()
      l.Add( TObjString('a') )
      l.Add( TObjString('b') )
      l.Add( TObjString('c') )
      l.Add( TObjString('d') )
      l.Add( TObjString('e') )
      l.Add( TObjString('f') )
      l.Add( TObjString('g') )
      l.Add( TObjString('h') )
      l.Add( TObjString('i') )
      l.Add( TObjString('j') )

      self.assertEqual( len(l), 10 )
      self.assertEqual( l[3], 'd' )
      self.assertEqual( l[-1], 'j' )
      self.assertRaises( IndexError, l.__getitem__,  20 )
      self.assertRaises( IndexError, l.__getitem__, -20 )

      self.assertEqual( list(l), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] )

      l[3] = TObjString('z')
      self.assertEqual( list(l), ['a', 'b', 'c', 'z', 'e', 'f', 'g', 'h', 'i', 'j'] )

      del l[2]
      self.assertEqual( list(l), ['a', 'b', 'z', 'e', 'f', 'g', 'h', 'i', 'j'] )

      self.assert_( TObjString('b') in l )
      self.assert_( not TObjString('x') in l )

      self.assertEqual( list(l[2:6]),   ['z', 'e', 'f', 'g'] )
      self.assertEqual( list(l[2:6:2]), ['z', 'f'] )
      self.assertEqual( list(l[-5:-2]), ['f', 'g', 'h'] )
      self.assertEqual( list(l[7:]),    ['i', 'j'] )
      self.assertEqual( list(l[:3]),    ['a', 'b', 'z'] )

      del l[2:4]
      self.assertEqual( list(l), ['a', 'b', 'f', 'g', 'h', 'i', 'j'] )

      l[2:5] = [ TObjString('1'), TObjString('2') ]
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j'] )

      l[6:6] = [ TObjString('3') ]
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j', '3'] )

      l.append( TObjString('4') )
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j', '3', '4'] )

      l.extend( [ TObjString('5'), TObjString('j') ] )
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j', '3', '4', '5', 'j'] )
      self.assertEqual( l.count( 'b' ), 1 )
      self.assertEqual( l.count( 'j' ), 2 )
      self.assertEqual( l.count( 'x' ), 0 )

      self.assertEqual( l.index( TObjString( 'i' ) ), 4 )
      self.assertRaises( ValueError, l.index, TObjString( 'x' ) )

      l.insert(  3, TObjString('6') )
      l.insert( 20, TObjString('7') )
      l.insert( -1, TObjString('8') )
      self.assertEqual( list(l), ['8', 'a', 'b', '1', '6', '2', 'i', 'j', '3', '4', '5', 'j', '7'] )
      self.assertEqual( l.pop(), '7' )
      self.assertEqual( l.pop(3), '1' )
      self.assertEqual( list(l), ['8', 'a', 'b', '6', '2', 'i', 'j', '3', '4', '5', 'j'] )

      l.remove( TObjString( 'j' ) )
      l.remove( TObjString( '3' ) )

      self.assertRaises( ValueError, l.remove, TObjString( 'x' ) )
      self.assertEqual( list(l), ['8', 'a', 'b', '6', '2', 'i', '4', '5', 'j'] )

      l.reverse()
      self.assertEqual( list(l), ['j', '5', '4', 'i', '2', '6', 'b', 'a', '8'] )

      l.sort()
      self.assertEqual( list(l), ['2', '4', '5', '6', '8', 'a', 'b', 'i', 'j'] )

      l.sort( lambda a, b: cmp(b.GetName(),a.GetName()) )
      self.assertEqual( list(l), ['j', 'i', 'b', 'a', '8', '6', '5', '4', '2'] )

      l2 = l[:3]
      self.assertEqual( list(l2 * 3), ['j', 'i', 'b', 'j', 'i', 'b', 'j', 'i', 'b'] )
      self.assertEqual( list(3 * l2), ['j', 'i', 'b', 'j', 'i', 'b', 'j', 'i', 'b'] )

      l2 *= 3
      self.assertEqual( list(l2), ['j', 'i', 'b', 'j', 'i', 'b', 'j', 'i', 'b'] )

      l2 = l[:3]
      l3 = l[6:8]
      self.assertEqual( list(l2+l3), ['j', 'i', 'b', '5', '4'] )

      i = iter(l2)
      self.assertEqual( i.next(), 'j' )
      self.assertEqual( i.next(), 'i' )
      self.assertEqual( i.next(), 'b' )
      self.assertRaises( StopIteration, i.next )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
