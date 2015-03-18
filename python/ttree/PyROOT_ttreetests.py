# File: roottest/python/ttree/PyROOT_ttreetests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 10/13/06
# Last: 06/16/14

"""TTree reading/writing unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import *

__all__ = [
   'TTree1ReadWriteSimpleObjectsTestCase'
]

gROOT.LoadMacro( "TTreeTypes.C+" )


### Write/Read an std::vector to/from file ===================================
class TTree1ReadWriteSimpleObjectsTestCase( MyTestCase ):
   N, M = 5, 10
   fname, tname, ttitle = 'test.root', 'test', 'test tree'
   testnames = ['aap', 'noot', 'mies', 'zus', 'jet']

   def test01WriteStdVector( self ):
      """Test writing of a single branched TTree with an std::vector<double>"""

      f = TFile( self.fname, 'RECREATE' )
      t = TTree( self.tname, self.ttitle )
      v = std.vector( 'double' )()
      t.Branch( 'mydata', v.__class__.__name__, v )

      for i in range( self.N ):
         for j in range( self.M ):
            v.push_back( i*self.M+j )
         t.Fill()
         v.clear()
      f.Write()
      f.Close()

   def test02ReadStdVector( self ):
      """Test reading of a single branched TTree with an std::vector<double>"""

      f = TFile( self.fname )
      mytree = f.Get( self.tname )

      i = 0
      for event in mytree:
         for entry in mytree.mydata:
            self.assertEqual( i, int(entry) )
            i += 1
      self.assertEqual( i, self.N * self.M )

      f.Close()

   def test03WriteSomeDataObject( self ):
      """Test writing of a complex data object"""

      f = TFile( self.fname, 'RECREATE' )
      t = TTree( self.tname, self.ttitle )

      d = SomeDataObject()
      t.Branch( 'data', d );

      for i in range( self.N ):
         for j in range( self.M ):
            d.AddFloat( i*self.M+j )

         d.AddTuple( d.GetFloats() )

         t.Fill()

      f.Write()
      f.Close()


   def test04ReadSomeDataObject( self ):
      """Test reading of a complex data object"""

      f = TFile( self.fname )
      mytree = f.Get( self.tname )

      for event in mytree:
         i = 0
         for entry in event.data.GetFloats():
            self.assertEqual( i, int(entry) )
            i += 1

         for mytuple in event.data.GetTuples():
            i = 0
            for entry in mytuple:
               self.assertEqual( i, int(entry) )
               i += 1

      f.Close()

   def test05WriteSomeDataObjectBranched( self ):
      """Test writing of a complex object across different branches"""

      f = TFile( self.fname, 'RECREATE' )
      t = TTree( self.tname, self.ttitle )

      d = SomeDataStruct()

    # note: for p2.2, which has incomplete support of property types,
    # we need to keep a reference alive to the result of the property
    # call, or it will be deleted too soon; for later pythons, it is
    # safe to use d.Floats directly in the Branch() call
      fl = d.Floats
      t.Branch( 'floats', fl )
      t.Branch( 'nlabel', AddressOf( d, 'NLabel' ), 'NLabel/I' )
      t.Branch( 'label',  AddressOf( d, 'Label' ),  'Label/C' )

      for i in range( self.N ):
         for j in range( self.M ):
            d.Floats.push_back( i*self.M+j )

         d.NLabel = i
         d.Label  = '%d' % i

         t.Fill()

      f.Write()
      f.Close()

   def test06ReadSomeDataObjectBranched( self ):
      """Test reading of a complex object across different branches"""

      f = TFile( self.fname )
      mytree = f.Get( self.tname )

      for event in mytree:
         i, j = 0, 0
         for entry in event.floats:
            self.assertEqual( i, int(entry) )
            i += 1

         label = event.Label[0:event.Label.find('\0')]
         self.assertEqual( label, str(int(event.NLabel)) )

      f.Close()

   def test07WriteNonTObject( self ):
      """Test writing of a non-TObject derived instance"""

      f = TFile( self.fname, 'RECREATE' )

      myarray = TArrayI( 1 )
      f.WriteObject( myarray, 'myarray' )

      f.Close()

   def test08ReadNonTObject( self ):
      """Test reading of a non-TObject derived instance"""

      f = TFile( self.fname )

      #myarray = f.Get( 'myarray' )
      #self.assert_( isinstance( myarray, TArrayI ) )

      myarray = MakeNullPointer( TArrayI )
      f.GetObject( 'myarray', myarray )

      f.Close()

   def test09WriteBuiltinArray( self ):
      """Test writing of a builtin array"""

      f = TFile( self.fname, 'RECREATE' )

      CreateArrayTree()

      f.Write()
      f.Close()

   def test10ReadBuiltinArray( self ):
      """Test reading of a builtin array"""

      f = TFile( self.fname )

      t = f.Proto2Analyzed
      self.assertEqual( type(t), TTree )
      self.assertEqual( t.GetEntriesFast(), 1 )

      t.GetEntry( 0 )
      vals = [ -1 ,  -1 , 428 ,  0 ,  -1 ,
              167 ,   0 ,   0 ,  0 , 403 ,
               -1 ,  -1 , 270 , -1 ,   0 ,
               -1 , 408 ,   0 , -1 , 198 ]

      self.assertEqual( len(t.t0), 28 )

      for i in xrange(len(vals)):
         self.assertEqual( vals[i], t.t0[i] )

      f.Close()

   def test11WriteTObjArray( self ):
      """Test writing of a TObjArray"""

      f = TFile( self.fname, 'RECREATE' )
      t = TTree( self.tname, self.ttitle )
      o = TObjArray()
      t.Branch( 'mydata', o )

      nameds = [ TNamed( name, name ) for name in self.testnames ]
      for name in nameds:
         o.Add( name )
      self.assertEqual( len(o), len(self.testnames) )

      t.Fill()

      f.Write()
      f.Close()

   def test12ReadBuiltinArray( self ):
      """Test reading of a TObjArray"""

      f = TFile( self.fname )
      t = f.Get( self.tname )

      t.GetEntry( 0 )
      self.assertEqual( len(t.mydata), len(self.testnames) )
      for i in xrange(len(t.mydata)):
         self.assertEqual( t.mydata[i].GetName(), self.testnames[i] )

      f.Close()

   def test13WriteMisnamedLeaf( self ):
      """Test writing of an differently named leaf"""

      f = TFile( self.fname, 'RECREATE' )
      t = TTree( self.tname, self.ttitle )
      s = SomeDataStruct()
      t.Branch( 'cpu_packet_time', AddressOf(s, 'NLabel'), 'time/I' );

      for i in range(self.N):
         s.NLabel = i
         t.Fill()

      f.Write()
      f.Close()

   def test14ReadMisnamedLeaf( self ):
      """Test reading of an differently named leaf"""

      f = TFile( self.fname )
      t = f.Get( self.tname )

      val = 0
      for event in t:
         event.time == val             # from 'time/I' label
         event.cpu_packet_time == val  # from branch name
         val += 1

      f.Close()


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
