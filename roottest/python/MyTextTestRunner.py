# File: roottest/python/MyTextTestRunner.py
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 03/18/05
# Last: 03/18/05

import unittest


if hasattr( unittest, 'TextTestResult' ):
   class MyTextTestResult( unittest.TextTestResult ):
      def getDescription(self, test):
         return test.shortDescription()
else:
   class MyTextTestResult( object ):
      pass


class MyTextTestRunner( unittest.TextTestRunner ):
   resultclass = MyTextTestResult

   def run( self, test ):
      """Run the given test case or test suite."""

      result = self._makeResult()
      test( result )
      result.printErrors()
      self.stream.writeln( result.separator2 )
      run = result.testsRun
      self.stream.writeln()

      if not result.wasSuccessful():
         self.stream.write( "FAILED (" )
         failed, errored = map( len, ( result.failures, result.errors ) )
         if failed:
            self.stream.write( "failures=%d" % failed )
         if errored:
            if failed: self.stream.write( ", " )
            self.stream.write( "errors=%d" % errored )
         self.stream.writeln( ")" )
      else:
         self.stream.writeln( "OK" )
   
      return result
