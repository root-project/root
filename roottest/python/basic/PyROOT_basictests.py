# File: roottest/python/basic/PyROOT_basictests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 11/23/04
# Last: 06/09/15

"""Basic unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ROOT
from ROOT import gROOT, gApplication, gSystem, gInterpreter, gDirectory
from ROOT import TLorentzVector, TString, TList, TH1D, TObjArray, TVectorF, TObjString
from common import *

__all__ = [
   'Basic1ModuleTestCase',
   'Basic2SetupTestCase',
   'Basic5PythonizationTestCase',
]

def setup_module(mod):
    if not os.path.exists('SimpleClass.C'):
        os.chdir(os.path.dirname(__file__))

### basic module test cases ==================================================
class Basic1ModuleTestCase( MyTestCase ):
   def test1Import( self ):
      """Test import error handling"""

      def failImport():
         from ROOT import GatenKaas

      self.assertRaises( ImportError, failImport )


### basic functioning test cases =============================================
class Basic2SetupTestCase( MyTestCase ):
   def test1Globals( self ):
      """Test the availability of ROOT globals"""

      self.assertTrue( gROOT )
      self.assertTrue( gApplication )
      self.assertTrue( gSystem )
      self.assertTrue( gInterpreter )
      self.assertTrue( gDirectory )

   def test2AccessToGlobals( self ):
      """Test overwritability of ROOT globals"""

      import ROOT
      oldval = ROOT.gDebug

      ROOT.gDebug = -1
      self.assertTrue(gROOT.ProcessLine('gDebug == -1'))

      ROOT.gDebug = oldval

   def test3AccessToGlobalsFromROOT( self ):
      """Test creation and access of new ROOT globals"""

      import ROOT
      ROOT.gMyOwnGlobal = 3.1415

      proxy = gROOT.GetGlobal( 'gMyOwnGlobal', 1 )
      try:
         self.assertEqual( proxy.__get__( proxy ), 3.1415 )
      except AttributeError:
         # In the old PyROOT, if we try to bind a new global,
         # such global is defined on the C++ side too, but
         # only if its type is basic or string (see ROOT.py).
         # The new PyROOT will discontinue this feature.

         # Note that in the new PyROOT we can still define a
         # global in C++ and access/modify it from Python
         ROOT.gInterpreter.Declare("int gMyOwnGlobal2 = 1;")
         self.assertEqual(ROOT.gMyOwnGlobal2, 1)
         ROOT.gMyOwnGlobal2 = -1
         self.assertTrue(gROOT.ProcessLine('gMyOwnGlobal2 == -1'))

   def test4AutoLoading( self ):
      """Test auto-loading by retrieving a non-preloaded class"""

      t = TLorentzVector()
      self.assertTrue( isinstance( t, TLorentzVector ) )

   def test5MacroLoading( self ):
      """Test accessibility to macro classes"""
      gROOT.LoadMacro( 'SimpleClass.C' )

      self.assertTrue( issubclass( ROOT.SimpleClass, ROOT.TheBase ) )
      self.assertEqual( ROOT.SimpleClass, ROOT.SimpleClass_t )

      c = ROOT.SimpleClass()
      self.assertTrue( isinstance( c, ROOT.SimpleClass ) )
      self.assertEqual( c.fData, c.GetData() )

      c.SetData( 13 )
      self.assertEqual( c.fData, 13 )
      self.assertEqual( c.GetData(), 13 )


### basic extension features test cases ======================================
class Basic5PythonizationTestCase( MyTestCase ):
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
      """Test list/TList behavior and compatibility"""

      # A TList is non-owning. In order to fill the TList with in-place created
      # objects, we write this little helper to create a new TObjectString
      # whose lifetime is managed by a separate owning Python list.
      objects = []
      def make_obj_str(s):
          objects.append(TObjString(s))
          return objects[-1]

      l = TList()
      l.Add( make_obj_str('a') )
      l.Add( make_obj_str('b') )
      l.Add( make_obj_str('c') )
      l.Add( make_obj_str('d') )
      l.Add( make_obj_str('e') )
      l.Add( make_obj_str('f') )
      l.Add( make_obj_str('g') )
      l.Add( make_obj_str('h') )
      l.Add( make_obj_str('i') )
      l.Add( make_obj_str('j') )

      self.assertEqual( len(l), 10 )
      self.assertEqual( l[3], 'd' )
      self.assertEqual( l[-1], 'j' )
      self.assertRaises( IndexError, l.__getitem__,  20 )
      self.assertRaises( IndexError, l.__getitem__, -20 )

      self.assertEqual( list(l), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] )

      l[3] = make_obj_str('z')
      self.assertEqual( list(l), ['a', 'b', 'c', 'z', 'e', 'f', 'g', 'h', 'i', 'j'] )

      del l[2]
      self.assertEqual( list(l), ['a', 'b', 'z', 'e', 'f', 'g', 'h', 'i', 'j'] )

      self.assertTrue( make_obj_str('b') in l )
      self.assertTrue( not make_obj_str('x') in l )

      self.assertEqual( list(l[2:6]),   ['z', 'e', 'f', 'g'] )
      self.assertEqual( list(l[2:6:2]), ['z', 'f'] )
      self.assertEqual( list(l[-5:-2]), ['f', 'g', 'h'] )
      self.assertEqual( list(l[7:]),    ['i', 'j'] )
      self.assertEqual( list(l[:3]),    ['a', 'b', 'z'] )

      del l[2:4]
      self.assertEqual( list(l), ['a', 'b', 'f', 'g', 'h', 'i', 'j'] )

      l[2:5] = [ make_obj_str('1'), make_obj_str('2') ]
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j'] )

      l[6:6] = [ make_obj_str('3') ]
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j', '3'] )

      l.append( make_obj_str('4') )
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j', '3', '4'] )

      l.extend( [ make_obj_str('5'), make_obj_str('j') ] )
      self.assertEqual( list(l), ['a', 'b', '1', '2', 'i', 'j', '3', '4', '5', 'j'] )
      self.assertEqual( sum(1 for x in l if x == 'b'), 1 )
      self.assertEqual( sum(1 for x in l if x == 'j'), 2 )
      self.assertEqual( sum(1 for x in l if x == 'x'), 0 )

      self.assertEqual( l.index( make_obj_str( 'i' ) ), 4 )
      self.assertRaises( ValueError, l.index, make_obj_str( 'x' ) )

      l.insert(  3, make_obj_str('6') )
      l.insert( 20, make_obj_str('7') )
      l.insert( -1, make_obj_str('8') )
      # The pythonisation of TSeqCollection in experimental PyROOT mimics the
      # behaviour of the Python list, in this case for insert.
      # The Python list insert always inserts before the specified index, so if
      # -1 is specified, insert will place the new element right before the last
      # element of the list.
      self.assertEqual(list(l), ['a', 'b', '1', '6', '2', 'i', 'j', '3', '4', '5', 'j', '8', '7'])
      # Re-synchronize with current PyROOT's list
      l.insert(0, make_obj_str('8'))
      self.assertEqual(list(l), ['8', 'a', 'b', '1', '6', '2', 'i', 'j', '3', '4', '5', 'j', '8', '7'])
      l.pop(-2)
      self.assertEqual(list(l), ['8', 'a', 'b', '1', '6', '2', 'i', 'j', '3', '4', '5', 'j', '7'])
      self.assertEqual( l.pop(), '7' )
      self.assertEqual( l.pop(3), '1' )
      self.assertEqual( list(l), ['8', 'a', 'b', '6', '2', 'i', 'j', '3', '4', '5', 'j'] )

      l.remove( make_obj_str( 'j' ) )
      l.remove( make_obj_str( '3' ) )

      self.assertRaises( ValueError, l.remove, make_obj_str( 'x' ) )
      self.assertEqual( list(l), ['8', 'a', 'b', '6', '2', 'i', '4', '5', 'j'] )

      l.reverse()
      self.assertEqual( list(l), ['j', '5', '4', 'i', '2', '6', 'b', 'a', '8'] )

      l.sort()
      self.assertEqual( list(l), ['2', '4', '5', '6', '8', 'a', 'b', 'i', 'j'] )

      l.sort( key=TObjString.GetName )
      l.reverse()
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
      self.assertEqual( getattr( i, "__next__" )(), 'j' )
      self.assertEqual( getattr( i, "__next__" )(), 'i' )
      self.assertEqual( getattr( i, "__next__" )(), 'b' )
      self.assertRaises( StopIteration, getattr( i, "__next__" ) )

   def test3TVector( self ):
      """Test TVector2/3/T behavior"""

      import math

      N = 51

    # TVectorF is a typedef of floats
      v = TVectorF(N)
      for i in range( N ):
          v[i] = i*i

      for j in v:
         self.assertEqual( round( v[ int(math.sqrt(j)+0.5) ] - j, 5 ), 0 )

   def test4TObjArray( self ):
      """Test TObjArray iterator-based copying"""

      a = TObjArray()
      b = list( a )

      self.assertEqual( b, [] )

   def test5Hashing( self ):
      """C++ objects must be hashable"""

      a = TH1D("asd", "asd", 10, 0, 1)
      self.assertTrue( hash(a) )


## actual test run
if __name__ == '__main__':
   #import common
   result = run_pytest(__file__)
   sys.exit(result)
