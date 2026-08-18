# \file
# \ingroup tutorial_math
# \notebook -nodraw
#
# Example on how to use the multiroot-finder based on the GSL-algorithm(library).
# This script finds the root of the Rosenbrock system of equations:
# \f[
# f1(x,y) = a(1-x)
# \f]
# \f[
# f2(x,y) = b(y-x^2)
# \f]
# with:
# \f[
# a = 1, b=10
# \f]
#
# The MultiRootFinder is based on GSL, and it requires to have
# the MathMore library installed.
#
# Usage:
#
# ~~~{.py}
# IP[0]: %run exampleMultiRoot.py
# ~~~
#
# or
#
# ~~~{.py}
# IP[0]: %run exampleMultiRoot.py
# IP[1]: exampleMultiRoot(algoname="hybridS" ,printlevel=1 )
# ~~~
#
# where algoname stands for "algorithm name" which 
# no uses derivatives, like:
# hybridS (default) , hybrid, dnewton, broyden;
# and printlevel is an integer refering to print those 
# needed levels until the result is converged. 
# 
# Enjoy!
#
# \macro_output
# \macro_code
#
# \author Lorenzo Moneta
import ROOT
import ctypes

#RConfigure = 		 ROOT.RConfigure
TF2 = 		 ROOT.TF2
#TError = 		 ROOT.TError

#ifdef R__HAS_MATHMORE
Math = ROOT.Math
try:
   MultiRootFinder = Math.MultiRootFinder
#else
except :
   raise RuntimeError("libMathMore is not available")
#error libMathMore is not available - cannot run this tutorial
#endif

WrappedMultiTF1 = Math.WrappedMultiTF1


#types
nullptr = ROOT.nullptr
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t
Char_t  = ROOT.Char_t
c_double = ctypes.c_double

#types
def to_c(ls):
   return (c_double * len(ls) ) ( * ls)


# Example on how to use the multi-root finder based on GSL.
# It needs to use an algorithm not requiring its derivative,
# like: hybrids(default), hybrid, dnewton or broyden.


#void
def exampleMultiRoot( algo : Char_t = 0, printlevel : Int_t = 1) :

   global r
   r = ROOT.Math.MultiRootFinder(algo)

   #defining the functions
   # use Rosenbrock functions
   global f1, f2
   f1 = TF2("f1","[0]*(1-x)+[1]*y")
   f2 = TF2("f2","[0]*(y-x*x)")


   f1.SetParameters(1,0)
   f2.SetParameter(0,10)

   # wrap the functions
   global g1, g2
   g1 = ROOT.Math.WrappedMultiTF1(f1, 2)
   g2 = ROOT.Math.WrappedMultiTF1(f2, 2)

   r.AddFunction(g1)
   r.AddFunction(g2)
   r.SetPrintLevel(printlevel)
   
   # starting point
   x0 = [ -1,-1 ]
   c_x0 = to_c(x0)

   r.Solve( c_x0 )
   

if __name__ == "__main__":
  exampleMultiRoot() 
