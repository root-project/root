# File: roottest/python/cpp11/PyROOT_cpptests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 11/25/13
# Last: 11/26/13

"""C++11 language interface unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from ROOT import *
from common import *

__all__ = [
   'Cpp1Cpp11StandardClassesTestCase',
]

gROOT.LoadMacro( "Cpp11Features.C+" )


### C++11 language constructs test cases =====================================
class Cpp1Cpp11StandardClassesTestCase( MyTestCase ):
   def test01SharedPtr( self ):
      """Test usage and access of std::shared_ptr<>"""

      if not USECPP11:
         return

    # proper memory accounting
      self.assertEqual( MyCounterClass.counter, 0 )

      ptr1 = CreateMyCounterClass()
      self.assert_( not not ptr1 )
      self.assertEqual( MyCounterClass.counter, 1 )

      ptr2 = CreateMyCounterClass()
      self.assert_( not not ptr2 )
      self.assertEqual( MyCounterClass.counter, 2 )

      del ptr2, ptr1
      import gc; gc.collect()
      self.assertEqual( MyCounterClass.counter, 0 )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
