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
import ROOT
from ROOT import TH1F, gROOT, std
from common import *

__all__ = [
   'PickleWritingSimpleObjectsTestCase'
]

if not os.path.exists('PickleTypes.C'):
    os.chdir(os.path.dirname(__file__))

gROOT.LoadMacro( "PickleTypes.C+" )
SomeDataObject = ROOT.SomeDataObject


### Write various objects with the two pickle modules ========================
class PickleWritingSimpleObjectsTestCase( MyTestCase ):
   out1 = open( pclfn, 'wb' )      # names from common.py
   out2 = open( cpclfn, 'wb' )

   def test1WriteTObjectDerived( self ):
      """Test writing of a histogram into a pickle file"""

      h1 = TH1F( h1name, h1title, h1nbins, h1binl, h1binh )
      h1.FillRandom( 'gaus', h1entries )

      pickle.dump(  h1, self.out1 )
      cPickle.dump( h1, self.out2 )

   def test2WriteNonTObjectDerived( self ):
      """Test writing of an std::vector<double> into a pickle file"""

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

   def test4WriteROOTObjInNamespace( self ):
      """Test writing of a ROOT object in a namespace into a pickle file"""

      v = ROOT.ROOT.Math.SVector('double',2)(1,2)

      pickle.dump(  v, self.out1 )
      cPickle.dump( v, self.out2 )

   def test5WriteCustomTypes( self ):
      """Test writing PyROOT custom types"""

      # Cppyy's Long and Double will be deprecated in favour of
      # ctypes.c_long and ctypes.c_double, respectively
      # https://bitbucket.org/wlav/cppyy/issues/101
      import ctypes

      o = [ctypes.c_long(123), ctypes.c_double(123.123)]

      pickle.dump(  o, self.out1, protocol = 2 )
      cPickle.dump( o, self.out2, protocol = 2 )

   def test6WriteCustomTypes( self ):
      """[ROOT-10810] Test writing a RooDataSet with weights"""

      if os.environ.get('ROOFIT') == 'False':
          self.skipTest("ROOT was built without RooFit")

      var = ROOT.RooRealVar('var' ,'variable',0,10)
      w = ROOT.RooRealVar('w' ,'weight',0,10)
      vs = ROOT.RooArgSet ( var , w )
      ds = ROOT.RooDataSet('data', '', vs, ROOT.RooFit.WeightVar(w))

      # Only fails with tree storage
      ds.convertToTreeStore()
      for i in range ( 10 ) :
           var.setVal(i)
           ds.add(vs, i+0.1)

      pickle.dump(ds, self.out1, protocol = 2) ## <--- segmentation fault
      cPickle.dump(ds,self.out2, protocol = 2)

   def tearDown( self ):
      self.out1.flush()
      self.out2.flush()

## actual test run
if __name__ == '__main__':
   sys.path.append(os.path.dirname(os.path.dirname(__file__)))
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
