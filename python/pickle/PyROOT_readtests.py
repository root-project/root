# File: roottest/python/pickle/PyROOT_readingtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 04/16/08
# Last: 10/22/10

"""Pickle writing unit tests for PyROOT package."""

import os, sys, unittest, ctypes
try:
   import pickle, cPickle
except ImportError:
   import pickle as cPickle            # p3
import ROOT
from ROOT import gROOT, TBufferFile, TH1F, TBuffer, std
from common import *

__all__ = [
   'PickleReadingSimpleObjectsTestCase',
   'PickleReadingComplicationsTestCase'
]

if not os.path.exists('PickleTypes.C'):
    os.chdir(os.path.dirname(__file__))

gROOT.LoadMacro( "PickleTypes.C+" )
SomeCountedClass = ROOT.SomeCountedClass
SomeDataObject = ROOT.SomeDataObject


### Read various objects with the two pickle modules =========================
class PickleReadingSimpleObjectsTestCase( MyTestCase ):
   in1 = open( pclfn, 'rb' )       # names from common.py
   in2 = open( cpclfn, 'rb' )

 # note that the order of these tests have to match the writing order (for
 # simple indexing, shelve should have been used instead); this also means
 # that if reading of one test fails, everything downstream fails as well
   def test1ReadTObjectDerived( self ):
      """Test reading of a histogram from a pickle file"""

      def __doh1test( self, h1 ):
         self.assertEqual( h1.__class__, TH1F )
         self.assertEqual( h1.GetName(),     h1name )
         self.assertEqual( h1.GetTitle(),    h1title )
         self.assertEqual( h1.GetNbinsX(),   h1nbins )

      h1 = pickle.load( self.in1 )
      __doh1test( self, h1 )

      h1 = cPickle.load( self.in2 )
      __doh1test( self, h1 )

   def test2ReadNonTObjectDerived( self ):
      """Test reading of an std::vector<double> from a pickle file"""

      def __dovtest( self, v ):
         self.assertEqual( v.__class__, std.vector( 'double' ) )
         self.assertEqual( v.size(), Nvec )

         for i in range( Nvec ):
            self.assertEqual( v[i], i*i )

      v = pickle.load( self.in1 )
      __dovtest( self, v )

      v = cPickle.load( self.in2 )
      __dovtest( self, v )

   def test3ReadSomeDataObject( self ):
      """Test reading of a user-defined object from a pickle file"""

      def __dodtest( self, d ):
         self.assertEqual( d.__class__, SomeDataObject )

         i = 0
         for entry in d.GetFloats():
            self.assertEqual( i, int(entry) )
            i += 1

         for mytuple in d.GetTuples():
            i = 0
            for entry in mytuple:
               self.assertEqual( i, int(entry) )
               i += 1

      d = pickle.load( self.in1 )
      __dodtest( self, d )

      d = cPickle.load( self.in2 )
      __dodtest( self, d )

   def test4ReadROOTObjInNamespace( self ):
      """Test reading of a ROOT object in a namespace from a pickle file"""

      def __doftest( self, d ):
         self.assertEqual( d.__class__, ROOT.ROOT.Math.SVector('double',2) )
         self.assertEqual( d.At(0) , 1 )
         self.assertEqual( d.At(1) , 2 )


      d = pickle.load( self.in1 )
      __doftest( self, d )

      d = cPickle.load( self.in2 )
      __doftest( self, d )

   def test5ReadCustomTypes( self ):
      """Test reading PyROOT custom types"""

      legacy_pyroot = os.environ.get('LEGACY_PYROOT') == 'True'

      p = pickle.load( self.in1 )
      cp = cPickle.load( self.in2 )

      if not legacy_pyroot:
         # Cppyy's Long and Double will be deprecated in favour of
         # ctypes.c_long and ctypes.c_double, respectively
         # https://bitbucket.org/wlav/cppyy/issues/101
         import ctypes

         proto = [ctypes.c_long(123), ctypes.c_double(123.123)]

         for e1, e2 in zip(p, proto):
            self.assertEqual(e1.value, e2.value)

         for e1, e2 in zip(cp, proto):
            self.assertEqual(e1.value, e2.value)

      else:
         proto = [ROOT.Long(123), ROOT.Double(123.123)]

         self.assertEqual(p, [123, 123.123])
         self.assertEqual(cp, [123, 123.123])

   def test6ReadCustomTypes( self ):
      """[ROOT-10810] Test reading a RooDataSet with weights"""
      ROOT.gEnv.SetValue("RooFit.Banner", 0)
      ds = pickle.load( self.in1 )
      dsc= cPickle.load( self.in2 )
      self.assertEqual(ds.get(1)['var'].getVal(), 1)
      self.assertEqual(ds.weight(), 1.1)
      self.assertEqual(dsc.get(2)['var'].getVal(), 2)
      self.assertEqual(dsc.weight(), 2.1)

### Pretend-write and read back objects that gave complications ==============
class PickleReadingComplicationsTestCase( MyTestCase ):

   def test1RefCountCheck( self ):
      """Test reference counting of pickled object"""

      self.assertEqual( SomeCountedClass.s_counter, 0 )
      c1 = SomeCountedClass();
      self.assertEqual( SomeCountedClass.s_counter, 1 )

      c2 = pickle.loads( pickle.dumps( c1 ) )
      self.assertEqual( SomeCountedClass.s_counter, 2 )
      del c2, c1
      self.assertEqual( SomeCountedClass.s_counter, 0 )

   def test2TBufferCheck( self ):
      """Test that a TBufferFile can be pickled"""

    # the following does not assert anything, but if there is a failure, the
    # ROOT I/O layer will print an error message
      f1 = TBufferFile( TBuffer.kWrite )
      f2 = pickle.loads( pickle.dumps( f1 ) )

   def test3PickleFacadeCheck(self):
      """Test serialization of the ROOT Python module.

      This needs a custom __reduce__ method defined in the ROOTFacade class.
      """

      def get_root_facade():
         return ROOT

      facade = pickle.loads(pickle.dumps(get_root_facade()))

      # Check attributes of the unserialized facade
      self.assertEqual(facade.__name__, ROOT.__name__)
      self.assertEqual(facade.__file__, ROOT.__file__)


## actual test run
if __name__ == '__main__':
   sys.path.append(os.path.dirname(os.path.dirname(__file__)))
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
