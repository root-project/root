# File: roottest/python/common.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 09/24/10
# Last: 09/30/10

__all__ = [ 'pylong', 'maxvalue', 'MyTestCase' ]

import os, sys, unittest

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
