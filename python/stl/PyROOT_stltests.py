# File: roottest/python/stl/PyROOT_stltests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 10/25/05
# Last: 10/25/05

"""STL unit tests for PyROOT package."""

import os, sys, unittest
from ROOT import *

__all__ = [
   'STL1VectorTestCase',
   'STL2ListTestCase',
   'STL3MapTestCase'
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

   def test2GeneratedVectorType( self ):
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


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
