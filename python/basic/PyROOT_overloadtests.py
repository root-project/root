# File: roottest/python/basic/PyROOT_overloadtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 04/15/05
# Last: 04/15/05

"""Overload unit tests for PyROOT package."""

import os, sys, unittest
from array import array
from ROOT import *

__all__ = [
   'Overloads1ClassTestCase',
   'Overloads2TMathTestCase'
]

gROOT.LoadMacro( "Overloads.C+" )


### class/array overloaded functions =========================================
class Overloads1ClassArrayTestCase( unittest.TestCase ):
   def test1ClassOverloads( self ):
      """Test functions overloaded on different ROOT clases"""

      self.assertEqual( MyC().GetInt( MyA() ), 42 )
      self.assertEqual( MyC().GetInt( MyB() ), 13 )
      self.assertEqual( MyD().GetInt( MyA() ), 42 )
      self.assertEqual( MyD().GetInt( MyB() ), 13 )

   def test2ArrayOverloads( self ):
      """Test functions overloaded on different arrays"""

      ai = array( 'i', [ 525252 ] )
      self.assertEqual( MyC().GetInt( ai ), 525252 )
      self.assertEqual( MyD().GetInt( ai ), 525252 )

      ah = array( 'h', [ 25 ] )
      self.assertEqual( MyC().GetInt( ai ), 525252 )
      self.assertEqual( MyD().GetInt( ai ), 525252 )


### basic functioning test cases =============================================
class Overloads2TMathTestCase( unittest.TestCase ):
   def test1MeanOverloads( self ):
      """Test overloads using TMath::Mean(), TMath::Median"""

      numbers = [ 8, 2, 4, 2, 4, 2, 4, 4, 1, 5, 6, 3, 7 ]
      mean, median = 4.0, 4.0

      af = array( 'f', numbers )
      self.assertEqual( round( TMath.Mean( len(af), af ) - mean, 5 ), 0 )
      self.assertEqual( round( TMath.Median( len(af), af ) - median, 5 ), 0 )

      ad = array( 'd', numbers )
      self.assertEqual( round( TMath.Mean( len(ad), ad ) - mean, 8), 0 )
      self.assertEqual( round( TMath.Median( len(ad), ad ) - median, 8), 0 )

      ai = array( 'i', numbers )
      self.assertEqual( round( TMath.Mean( len(ai), ai ) - mean, 8), 0 )
      self.assertEqual( round( TMath.Median( len(ai), ai ) - median, 8), 0 )

      ah = array( 'h', numbers )
      self.assertEqual( round( TMath.Mean( len(ah), ah ) - mean, 8), 0 )
      self.assertEqual( round( TMath.Median( len(ah), ah ) - median, 8), 0 )

      al = array( 'l', numbers )
      self.assertEqual( round( TMath.Mean( len(al), al ) - mean, 8), 0 )
      self.assertEqual( round( TMath.Median( len(al), al ) - median, 8), 0 )

    # this one should fail because there's no TMath::Mean( Long64_t, ULong_t* )
      aL = array( 'L', numbers )
      self.assertRaises( TypeError, TMath.Mean, len(aL), aL )


## actual test run
if __name__ == '__main__':
   sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
