# File: roottest/python/stl/PyROOT_stltests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 10/25/05
# Last: 11/22/06

"""STL unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'STL1VectorTestCase',
   'STL2ListTestCase',
   'STL3MapTestCase',
   'STL4STLLikeClassTestCase',
   'STL5StringHandlingTestCase'
]

gROOT.LoadMacro( "StlTypes.C+" )


### STL vector test case =====================================================
class STL1VectorTestCase( unittest.TestCase ):
   N = 13

   def test1BuiltinVectorType( self ):
      """Test access to a vector<int> (part of cintdlls)"""

      a = std.vector( int )( self.N )
      for i in range(self.N):
         a[i] = i
         self.assertEqual( a[i], i )

      self.assertEqual( len(a), self.N )

   def test2BuiltinVectorType( self ):
      """Test access to a vector<double> (part of cintdlls)"""

      a = std.vector( 'double' )()
      for i in range(self.N):
         a.push_back( float(i) )
         self.assertEqual( a[i], i )

      self.assertEqual( len(a), self.N )

   def test3GeneratedVectorType( self ):
      """Test access to a ACLiC generated vector type"""

      a = std.vector( JustAClass )()
      self.assert_( hasattr( a, 'size' ) )
      self.assert_( hasattr( a, 'push_back' ) )
      self.assert_( hasattr( a, '__getitem__' ) )
      self.assert_( hasattr( a, 'begin' ) )
      self.assert_( hasattr( a, 'end' ) )

      for i in range(self.N):
         a.push_back( JustAClass() )
         a[i].m_i = i
         self.assertEqual( a[i].m_i, i )

      self.assertEqual( len(a), self.N )


### STL list test case =======================================================
class STL2ListTestCase( unittest.TestCase ):
   N = 13

   def test1BuiltinListType( self ):
      """Test access to a list<int> (part of cintdlls)"""

      a = std.list( int )()
      for i in range(self.N):
         a.push_back( i )

      self.assertEqual( len(a), self.N )
      self.failUnless( 11 in a )

      ll = list(a)
      for i in range(self.N):
         self.assertEqual( ll[i], i )


### STL map test case ========================================================
class STL3MapTestCase( unittest.TestCase ):
   N = 13

   def test1BuiltinMapType( self ):
      """Test access to a map<int,int> (part of cintdlls)"""

      a = std.map( int, int )()
      for i in range(self.N):
         a[i] = i
         self.assertEqual( a[i], i )

      self.assertEqual( len(a), self.N )


### Protocol mapping for an STL like class ===================================
class STL4STLLikeClassTestCase( unittest.TestCase ):
   def test1STLLikeClassIterators( self ):
      """Test the iterator protocol mapping for an STL like class"""

      a = STLLikeClass( int )()
      for i in a:
         pass

      self.assertEqual( i, 3 )


### String handling ==========================================================
class STL5StringHandlingTestCase( unittest.TestCase ):
   def test1StringArgumentPassing( self ):
      """Test mapping of python strings and std::string"""

      c, s = StringyClass(), std.string( "test1" )

    # pass through const std::string&
      c.SetString1( s )
      self.assertEqual( c.GetString1(), s )

      c.SetString1( "test2" )
      self.assertEqual( c.GetString1(), "test2" )

    # pass through std::string (by value)
      s = std.string( "test3" )
      c.SetString2( s )
      self.assertEqual( c.GetString1(), s )

      c.SetString2( "test4" )
      self.assertEqual( c.GetString1(), "test4" )

    # getting through std::string&
      s2 = std.string()
      c.GetString2( s2 )
      self.assertEqual( s2, "test4" )

      self.assertRaises( TypeError, c.GetString2, "temp string" )

   def test2StringDataAccess( self ):
      """Test access to std::string object data members"""

      c, s = StringyClass(), std.string( "test string" )

      c.m_string = s
      self.assertEqual( c.m_string, s )
      self.assertEqual( c.GetString1(), s )

      c.m_string = "another test"
      self.assertEqual( c.m_string, "another test" )
      self.assertEqual( c.GetString1(), "another test" )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
