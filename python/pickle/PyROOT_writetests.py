# File: roottest/python/pickle/PyROOT_writetests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 04/16/08
# Last: 10/22/10

"""Pickle writing unit tests for PyROOT package."""

import os, sys, unittest
try:
   import pickle, cPickle
except ImportError:
   import pickle as cPickle
from ROOT import *
from common import *

__all__ = [
   'PickleWritingSimpleObjectsTestCase'
]

gROOT.LoadMacro( "PickleTypes.C+" )


### Write various objects with the two pickle modules ========================
class PickleWritingSimpleObjectsTestCase( MyTestCase ):
   out1 = open( pclfn, 'wb' )      # names from common.py
   out2 = open( cpclfn, 'wb' )

   def test1WriteTObjectDerived( self ):
      """Test writing of a histogram into a pickle file"""

      if FIXCLING:
         return

      h1 = TH1F( h1name, h1title, h1nbins, h1binl, h1binh )
      h1.FillRandom( 'gaus', h1entries )

      pickle.dump(  h1, self.out1 )
      cPickle.dump( h1, self.out2 )

   def test2WriteNonTObjectDerived( self ):
      """Test writing of an std::vector<double> into a pickle file"""

      if FIXCLING:
         return

      v = std.vector( 'double' )()

      for i in range( Nvec ):
         v.push_back( i*i )

      pickle.dump(  v, self.out1 )
      cPickle.dump( v, self.out2 )

   def test3WriteSomeDataObject( self ):
      """Test writing of a user-defined object into a pickle file"""

      d = SomeDataObject()

      for i in range( Nvec ):
         for j in range( Mvec ):
            d.AddFloat( i*Mvec+j )

         d.AddTuple( d.GetFloats() )

      pickle.dump(  d, self.out1 )
      cPickle.dump( d, self.out2 )

   def tearDown( self ):
      self.out1.flush()
      self.out2.flush()


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
