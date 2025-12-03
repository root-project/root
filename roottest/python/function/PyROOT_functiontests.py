# File: roottest/python/function/PyROOT_functiontests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 11/24/04
# Last: 04/27/16

"""Unit tests for PyROOT python/TF1 function interactions."""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math import exp
import ROOT
from ROOT import TF1, TH1F, TMinuit, gROOT
from common import *
import ctypes

__all__ = [
   'Func1CallFunctionTestCase',
   'Func2FitFunctionTestCase',
   'Func3GlobalCppFunctionTestCase',
   'Func4GlobalCppFunctionAsMethodTestCase',
   'Func5MinuitTestCase'
]

if not os.path.exists('InstallableFunction.C'):
    os.chdir(os.path.dirname(__file__))

# needs to be early to prevent "ifunc_table overflow!"
gROOT.LoadMacro( "InstallableFunction.C+" )


### helpers for general test cases -------------------------------------------
def identity( x, par = None ):
   return x[0]

class Linear:
   def __call__( self, x, par ):
      return par[0] + x[0]*par[1]

def pygaus( x, par ):
    arg1 = 0
    scale1 =0
    ddx = 0.01

    if (par[2] != 0.0):
        arg1 = (x[0]-par[1])/par[2]
        scale1 = (ddx*0.39894228)/par[2]
        h1 = par[0]/(1+par[3])

        gauss = h1*scale1*exp(-0.5*arg1*arg1)
    else:
        gauss = 0.
    return gauss


### fit function and helper for minuit test case -----------------------------
def fcn( npar, gin, f, par, iflag ):
   global ncount
   nbins = 5

 # calculate chisquare
   chisq, delta = 0., 0.
   for i in range(nbins):
      delta  = (z[i]-func(x[i],y[i],par))/errorz[i]
      chisq += delta*delta

   # In the new Cppyy, f is a ctypes.c_double (see ROOT-10029).
   # Thus, the assignment needs to be done to its value attribute
   f.value = chisq
   ncount += 1

def func( x, y, par ):
   value = ( (par[0]*par[0])/(x*x) -1 ) / ( par[1]+par[2]*y-par[3]*y*y)
   return value


### data for minuit test -----------------------------------------------------
from array import array

Error = 0;
z = array( 'f', [ 1., 0.96, 0.89, 0.85, 0.78 ] )
errorz = array( 'f', 5*[0.01] )

x = array( 'f', [ 1.5751, 1.5825,  1.6069,  1.6339,   1.6706  ] )
y = array( 'f', [ 1.0642, 0.97685, 1.13168, 1.128654, 1.44016 ] )

ncount = 0


### basic function test cases ================================================
class Func1CallFunctionTestCase( MyTestCase ):
   def test1GlobalFunction( self ):
      """Test calling of a python global function"""

      f = TF1( "pyf1", identity, -1., 1., 0 )

      self.assertEqual( f.Eval(  0.5 ),   0.5 )
      self.assertEqual( f.Eval( -10. ), -10.  )
      self.assertEqual( f.Eval(  1.  ),   1.  )

    # check proper propagation of default value
      f = TF1( "pyf1d", identity, -1., 1. )

      self.assertEqual( f.Eval(  0.5 ),   0.5 )

   def test2CallableObject( self ):
      """Test calling of a python callable object"""

      pycal = Linear()
      f = TF1( "pyf2", pycal, -1., 1., 2 )
      f.SetParameters( 5., 2. )

      self.assertEqual( f.Eval( -0.1 ), 4.8 )
      self.assertEqual( f.Eval(  1.3 ), 7.6 )


### fitting with functions ===================================================
class Func2FitFunctionTestCase( MyTestCase ):
   def test1FitGaussian( self ):
      """Test fitting with a python global function"""

      f = TF1( 'pygaus', pygaus, -4, 4, 4 )
      f.SetParameters( 600, 0.43, 0.35, 600 )

      h = TH1F( "h", "test", 100, -4, 4 )
      h.FillRandom( "gaus", 200000 )
      h.Fit( f, "0Q" )

      self.assertEqual( f.GetNDF(), 96 )
      result = f.GetParameters()
      self.assertEqual( round( result[1] - 0., 1), 0 )  # mean
      self.assertEqual( round( result[2] - 1., 1), 0 )  # s.d.


### calling a global function ================================================
class Func3GlobalCppFunctionTestCase( MyTestCase ):
   def test1CallGlobalCppFunction( self ):
      """Test calling of an interpreted C++ global function"""

      gROOT.LoadMacro( "GlobalFunction.C" )
      InterpDivideByTwo = ROOT.InterpDivideByTwo
      

      self.assertEqual( round( InterpDivideByTwo( 4. ) - 4./2., 8), 0 )
      self.assertEqual( round( InterpDivideByTwo( 7. ) - 7./2., 8), 0 )

   def test2CallNameSpacedGlobalFunction( self ):
      """Test calling of an interpreted C++ namespaced global function"""

      InterpMyNameSpace = ROOT.InterpMyNameSpace

      self.assertEqual( round( InterpMyNameSpace.InterpNSDivideByTwo( 4. ) - 4./2., 8), 0 )
      self.assertEqual( round( InterpMyNameSpace.InterpNSDivideByTwo( 7. ) - 7./2., 8), 0 )

   def test3CallGlobalCppFunction( self ):
      """Test calling of a compiled C++ global function"""

      gROOT.LoadMacro( "GlobalFunction2.C+" )
      DivideByTwo = ROOT.DivideByTwo

      self.assertEqual( round( DivideByTwo( 4. ) - 4./2., 8), 0 )
      self.assertEqual( round( DivideByTwo( 7. ) - 7./2., 8), 0 )

   def test4CallNameSpacedGlobalFunction( self ):
      """Test calling of a compiled C++ namespaced global function"""

    # functions come in from GlobalFunction2.C, loaded in previous test3
      MyNameSpace = ROOT.MyNameSpace

      self.assertEqual( round( MyNameSpace.NSDivideByTwo( 4. ) - 4./2., 8), 0 )
      self.assertEqual( round( MyNameSpace.NSDivideByTwo( 7. ) - 7./2., 8), 0 )

   def test5CallAnotherNameSpacedGlobalFunction( self ):
      """Test namespace update after adding a global function"""

      gROOT.LoadMacro( "GlobalFunction3.C+" )
      MyNameSpace = ROOT.MyNameSpace

      self.assertEqual( round( MyNameSpace.NSDivideByTwo_v2( 4. ) - 4./2., 8), 0 )
      self.assertEqual( round( MyNameSpace.NSDivideByTwo_v2( 7. ) - 7./2., 8), 0 )


### using a global function as python class member ===========================
class Func4GlobalCppFunctionAsMethodTestCase( MyTestCase ):
   def test1InstallAndCallGlobalCppFunctionAsPythonMethod( self ):
      """Test installing and calling global C++ function as python method"""
      
      InstallableFunc = ROOT.InstallableFunc
      FuncLess = ROOT.FuncLess

      FuncLess.InstallableFunc = InstallableFunc

      a = FuncLess( 1234 )
      self.assertEqual( a.m_int, a.InstallableFunc().m_int );

   def test2InstallAndCallGlobalCppFunctionAsPythonMethod( self ):
      """Test installing and calling namespaced C++ function as python method"""
      
      FuncLess = ROOT.FuncLess
      FunctionNS = ROOT.FunctionNS

      FuncLess.InstallableFunc2 = FunctionNS.InstallableFunc

      a = FuncLess( 1234 )
      self.assertEqual( a.m_int, a.InstallableFunc2().m_int );


### test minuit callback functionality and fit results =======================
class Func5MinuitTestCase( MyTestCase ):
   def test1MinuitFit( self ):
      """Test minuit callback and fit"""

    # setup minuit and callback
      gMinuit = TMinuit(5)
      gMinuit.SetPrintLevel( -1 )            # quiet
      gMinuit.SetGraphicsMode( ROOT.kFALSE )
      gMinuit.SetFCN( fcn )

      arglist = array( 'd', 10*[0.] )
      ierflg = ctypes.c_int()

      arglist[0] = 1
      gMinuit.mnexcm( "SET ERR", arglist, 1, ierflg )

    # set starting values and step sizes for parameters
      vstart = array( 'd', [ 3,  1,  0.1,  0.01  ] )
      step   = array( 'd', [ 0.1, 0.1, 0.01, 0.001 ] )
      gMinuit.mnparm( 0, "a1", vstart[0], step[0], 0, 0, ierflg )
      gMinuit.mnparm( 1, "a2", vstart[1], step[1], 0, 0, ierflg )
      gMinuit.mnparm( 2, "a3", vstart[2], step[2], 0, 0, ierflg )
      gMinuit.mnparm( 3, "a4", vstart[3], step[3], 0, 0, ierflg )

    # now ready for minimization step
      arglist[0] = 500
      arglist[1] = 1.
      gMinuit.mnexcm( "MIGRAD", arglist, 2, ierflg )

    # verify results
      Double = ctypes.c_double
      amin, edm, errdef = Double(), Double(), Double()
      nvpar, nparx, icstat = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
      gMinuit.mnstat( amin, edm, errdef, nvpar, nparx, icstat )
    # gMinuit.mnprin( 3, amin )

      nvpar, nparx, icstat = map(lambda x: x.value, [nvpar, nparx, icstat])
      self.assertEqual( nvpar, 4 )
      self.assertEqual( nparx, 4 )

    # success means that full covariance matrix is available (icstat==3)
      self.assertEqual( icstat, 3 )

    # check results (somewhat debatable ... )
      par, err = Double(), Double()

      # ctypes.c_double requires the explicit retrieval of the inner value
      gMinuit.GetParameter( 0, par, err )
      self.assertEqual( round( par.value - 2.15, 2 ), 0. )
      self.assertEqual( round( err.value - 0.10, 2 ), 0. )

      gMinuit.GetParameter( 1, par, err )
      self.assertEqual( round( par.value - 0.81, 2 ), 0. )
      self.assertEqual( round( err.value - 0.25, 2 ), 0. )

      gMinuit.GetParameter( 2, par, err )
      self.assertEqual( round( par.value - 0.17, 2 ), 0. )
      self.assertEqual( round( err.value - 0.40, 2 ), 0. )

      gMinuit.GetParameter( 3, par, err )
      self.assertEqual( round( par.value - 0.10, 2 ), 0. )
      self.assertEqual( round( err.value - 0.16, 2 ), 0. )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
