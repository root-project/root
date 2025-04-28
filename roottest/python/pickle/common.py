# File: roottest/python/pickle/common.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 04/16/08
# Last: 09/24/10

import unittest

pclfn     = "PyROOT_test.pcl"
cpclfn    = "PyROOT_test.cpcl"

h1name    = 'h'
h1title   = 'test'
h1nbins   = 100
h1binl    = -4.
h1binh    = +4.
h1entries = 20000

Nvec      = 12
Mvec      =  7

class MyTestCase( unittest.TestCase ):
   def shortDescription( self ):
      desc = str(self)
      doc_first_line = None

      if self._testMethodDoc:
         doc_first_line = self._testMethodDoc.split("\n")[0].strip()
      if doc_first_line:
         desc = doc_first_line
      return desc
