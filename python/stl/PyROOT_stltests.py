1# File: roottest/python/stl/PyROOT_stltests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 10/25/05
# Last: 05/01/15

"""STL unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import *

__all__ = [
   'STL1VectorTestCase',
   'STL2ListTestCase',
   'STL3MapTestCase',
   'STL4STLLikeClassTestCase',
   'STL5StringHandlingTestCase',
   'STL6IteratorTestCase',
   'STL7StreamTestCase',
]

gROOT.LoadMacro( "StlTypes.C+" )


### STL vector test case =====================================================
class STL1VectorTestCase( MyTestCase ):
   N = 13

   def test1BuiltinVectorType( self ):
      """Test access to a vector<int> (part of cintdlls)"""

      a = std.vector( int )( self.N )
      self.assertEqual( len(a), self.N )

      for i in range(self.N):
         a[i] = i
         self.assertEqual( a[i], i )
         self.assertEqual( a.at(i), i )

      self.assertEqual( a.size(), self.N )
      self.assertEqual( len(a), self.N )

   def test2BuiltinVectorType( self ):
      """Test access to a vector<double> (part of cintdlls)"""

      a = std.vector( 'double' )()
      for i in range(self.N):
         a.push_back( i )
         self.assertEqual( a.size(), i+1 )
         self.assertEqual( a[i], i )
         self.assertEqual( a.at(i), i )

      self.assertEqual( a.size(), self.N )
      self.assertEqual( len(a), self.N )

   def test3GeneratedVectorType( self ):
      """Test access to a ACLiC generated vector type"""

      a = std.vector( JustAClass )()
      self.assert_( hasattr( a, 'size' ) )
      self.assert_( hasattr( a, 'push_back' ) )
      self.assert_( hasattr( a, '__getitem__' ) )
      self.assert_( hasattr( a, 'begin' ) )
      self.assert_( hasattr( a, 'end' ) )

      self.assertEqual( a.size(), 0 )

      for i in range(self.N):
         a.push_back( JustAClass() )
         a[i].m_i = i
         self.assertEqual( a[i].m_i, i )

      self.assertEqual( len(a), self.N )

   def test4EmptyVectorType( self ):
      """Test behavior of empty vector<int> (part of cintdlls)"""

      a = std.vector( int )()
      for arg in a:
         pass

   def test5PushbackIterablesWithIAdd( self ):
      """Test usage of += of iterable on push_back-able container"""

      a = std.vector( int )()
      a += [ 1, 2, 3 ]
      self.assertEqual( len(a), 3 )

      self.assertEqual( a[0], 1 )
      self.assertEqual( a[1], 2 )
      self.assertEqual( a[2], 3 )

      a += ( 4, 5, 6 )
      self.assertEqual( len(a), 6 )

      self.assertEqual( a[3], 4 )
      self.assertEqual( a[4], 5 )
      self.assertEqual( a[5], 6 )

      self.assertRaises( TypeError, a.__iadd__, ( 7, '8' ) )

   def test6VectorReturnDowncasting( self ):
      """Pointer returns of vector indexing must be down cast"""

      v = PR_Test.mkVect()
      self.assertEqual( type(v), std.vector( 'PR_Test::Base*' ) )
      self.assertEqual( len(v), 1 )
      self.assertEqual( type(v[0]), PR_Test.Derived )
      self.assertEqual( PR_Test.checkType(v[0]), PR_Test.checkType(PR_Test.Derived()) )

      p = PR_Test.check()
      self.assertEqual( type(p), PR_Test.Derived )
      self.assertEqual( PR_Test.checkType(p), PR_Test.checkType(PR_Test.Derived()) )


### STL list test case =======================================================
class STL2ListTestCase( MyTestCase ):
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

      for val in a:
         self.assertEqual( ll[ ll.index(val) ], val )

   def test2EmptyListType( self ):
      """Test behavior of empty list<int> (part of cintdlls)"""

      a = std.list( int )()
      for arg in a:
         pass


### STL map test case ========================================================
class STL3MapTestCase( MyTestCase ):
   N = 13

   def test01BuiltinMapType( self ):
      """Test access to a map<int,int> (part of cintdlls)"""

      a = std.map( int, int )()
      for i in range(self.N):
         a[i] = i
         self.assertEqual( a[i], i )

      self.assertEqual( len(a), self.N )

      for key, value in a:
         self.assertEqual( key, value )
      self.assertEqual( key,   self.N-1 )
      self.assertEqual( value, self.N-1 )

    # add a variation, just in case
      m = std.map( int, int )()
      for i in range(self.N):
         m[i] = i*i
         self.assertEqual( m[i], i*i )

      for key, value in m:
         self.assertEqual( key*key, value )
      self.assertEqual( key,   self.N-1 )
      self.assertEqual( value, (self.N-1)*(self.N-1) )

   def test02KeyedMapType( self ):
      """Test access to a map<std::string,int> (part of cintdlls)"""

      a = std.map( std.string, int )()
      for i in range(self.N):
         a[str(i)] = i
         self.assertEqual( a[str(i)], i )

      self.assertEqual( i, self.N-1 )
      self.assertEqual( len(a), self.N )

   def test03EmptyMapType( self ):
      """Test behavior of empty map<int,int> (part of cintdlls)"""

      m = std.map( int, int )()
      for key, value in m:
         pass

   def test04UnsignedvalueTypeMapTypes( self ):
      """Test assignability of maps with unsigned value types (not part of cintdlls)"""

      import math

      mui = std.map( str, 'unsigned int' )()
      mui[ 'one' ] = 1
      self.assertEqual( mui[ 'one' ], 1 )
      self.assertRaises( ValueError, mui.__setitem__, 'minus one', -1 )

    # UInt_t is always 32b, sys.maxint follows system int
      maxint32 = int(math.pow(2,31)-1)
      mui[ 'maxint' ] = maxint32 + 3
      self.assertEqual( mui[ 'maxint' ], maxint32 + 3 )

      mul = std.map( str, 'unsigned long' )()
      mul[ 'two' ] = 2
      self.assertEqual( mul[ 'two' ], 2 )
      mul[ 'maxint' ] = maxvalue + 3
      self.assertEqual( mul[ 'maxint' ], maxvalue + 3 )
      self.assertRaises( ValueError, mul.__setitem__, 'minus two', -2 )

   def test05FreshlyInstantiatedMapType( self ):
      """Instantiate a map from a newly defined class"""

      gInterpreter.Declare( 'template<typename T> struct Data { T fVal; };' )

      results = std.map( std.string, Data(int) )()
      d = Data(int)(); d.fVal = 42
      results[ 'summary' ] = d
      self.assertEqual( results.size(), 1 )
      for tag, data in results:
         self.assertEqual( data.fVal, 42 )


### Protocol mapping for an STL like class ===================================
class STL4STLLikeClassTestCase( MyTestCase ):
   def test1STLLikeClassIndexingOverloads( self ):
      """Test overloading of operator[] in STL like class"""

      a = STLLikeClass( int )()
      self.assertEqual( a[ "some string" ], 'string' )
      self.assertEqual( a[ 3.1415 ], 'double' )

   def test2STLLikeClassIterators( self ):
      """Test the iterator protocol mapping for an STL like class"""

      a = STLLikeClass( int )()
      for i in a:
         pass

      self.assertEqual( i, 3 )


### String handling ==========================================================
class STL5StringHandlingTestCase( MyTestCase ):
   def test1StringArgumentPassing( self ):
      """Test mapping of python strings and std::string"""

      c, s = StringyClass(), std.string( "test1" )

    # pass through const std::string&
      c.SetString1( s )
      self.assertEqual( type(c.GetString1()), str )
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

   def test3StringWithNullCharacter( self ):
      """Test that strings with NULL do not get truncated"""

      t0 = "aap\0noot"
      self.assertEqual( t0, "aap\0noot" )

      c, s = StringyClass(), std.string( t0, len(t0) )

      c.SetString1( s )
      self.assertEqual( t0, c.GetString1() )
      self.assertEqual( s, c.GetString1() )


### Iterator comparison ======================================================
class STL6IteratorComparisonTestCase( MyTestCase ):
   def __run_tests( self, container ):
      self.assertEqual( len(container), 1 )

      b1, e1 = container.begin(), container.end()
      b2, e2 = container.begin(), container.end()

      self.assert_( b1.__eq__( b2 ) )
      self.assert_( not b1.__ne__( b2 ) )
      if sys.hexversion < 0x3000000:
         self.assertEqual( cmp( b1, b2 ), 0 )

      self.assert_( e1.__eq__( e2 ) )
      self.assert_( not e1.__ne__( e2 ) )
      if sys.hexversion < 0x3000000:
         self.assertEqual( cmp( e1, e2 ), 0 )

      self.assert_( not b1.__eq__( e1 ) )
      self.assert_( b1.__ne__( e1 ) )
      if sys.hexversion < 0x3000000:
         self.assertNotEqual( cmp( b1, e1 ), 0 )

      b1.__preinc__()
      self.assert_( not b1.__eq__( b2 ) )
      self.assert_( b1.__eq__( e2 ) )
      if sys.hexversion < 0x3000000:
         self.assertNotEqual( cmp( b1, b2 ), 0 )
         self.assertEqual( cmp( b1, e1 ), 0 )
      self.assertNotEqual( b1, b2 )
      self.assertEqual( b1, e2 )

   def test1BuiltinVectorIterators( self ):
      """Test iterator comparison for vector"""

      v = std.vector( int )()
      v.resize( 1 )

      self.__run_tests( v )

   def test2BuiltinListIterators( self ):
      """Test iterator comparison for list"""

      l = std.list( int )()
      l.push_back( 1 )

      self.__run_tests( l )

   def test3BuiltinMapIterators( self ):
      """Test iterator comparison for map"""

      m = std.map( int, int )()
      m[1] = 1

      self.__run_tests( m )


### Stream usage =============================================================
class STL7StreamTestCase( MyTestCase ):
   def test1_PassStringStream( self ):
      """Pass stringstream through ostream&"""

      s = std.stringstream()
      o = StringStreamUser()

      o.fillStream( s )

      self.assertEqual( "StringStreamUser Says Hello!", s.str() )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
