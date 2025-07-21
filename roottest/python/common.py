# File: roottest/python/common.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 09/24/10
# Last: 04/04/14

__all__ = [ 'pylong', 'maxvalue', 'MyTestCase', 'run_pytest', 'FIXCLING' ]

import os, sys, unittest, warnings
import pytest


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

FIXCLING = '--fixcling' in sys.argv
if 'FIXCLING' in os.environ:
   FIXCLING = os.environ['FIXCLING'] == 'yes'


def run_pytest(test_file=None):
    # file to run, if any (search used otherwise)
    if '-i' in sys.argv:
        args = filter(lambda x: not x in (test_file, '-i'), sys.argv)
    else:
        args = ['--color=no']
    if test_file: args += [test_file]
    # actual test run
    return pytest.main(args)
