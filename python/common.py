# File: roottest/python/common.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 09/24/10
# Last: 04/04/14

__all__ = [ 'pylong', 'maxvalue', 'MyTestCase', 'run_pytest', 'FIXCLING' ]

import os, sys, unittest, warnings

# avoid conflict wuth the local pytest and py modules and the one from LCG pytools
warnings.filterwarnings('ignore', message=r'Module .*? is being added to sys\.path', append=True)

try:
   import pytest, pytest_cov
except:
   # pytest not found in the system, use local copy
   toplevel = os.path.dirname(os.path.realpath(__file__))
   sys.path.insert(0, os.path.join(toplevel, 'pytest'))
   import pytest, pytest_cov

if sys.hexversion >= 0x3000000:
   pylong = int
   maxvalue = sys.maxsize

   class MyTestCase( unittest.TestCase ):
      def shortDescription( self ):
         desc = str(self)
         doc_first_line = None

         if self._testMethodDoc:
            doc_first_line = self._testMethodDoc.split("\n")[0].strip()
         if doc_first_line:
            desc = doc_first_line
         return desc
else:
   pylong = long
   maxvalue = sys.maxint

   class MyTestCase( unittest.TestCase ):
      pass

FIXCLING = '--fixcling' in sys.argv
if 'FIXCLING' in os.environ:
   FIXCLING = os.environ['FIXCLING'] == 'yes'

def run_pytest(test_file=None):
    if '-i' in sys.argv:
        args = filter(lambda x: not x in (test_file, '-i'), sys.argv)
    elif test_file:
        args = test_file
    return pytest.main(args, plugins=[pytest_cov])
