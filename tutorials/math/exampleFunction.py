## \file
## \ingroup tutorial_math
## \notebook
## Example of using Python functions and input to numerical algorithm
## using the ROOT Functor class 
##
## \author Lorenzo Moneta

import ROOT
import array
try:
    import numpy as np
except:
    print("Failed to import numpy.")
    exit()

## example 1D function

def f(x):
   return x*x -1

func = ROOT.Math.Functor1D(f)

#example of using the integral

print("Use Functor1D for wrapping one-dimensional function and compute integral of f(x) = x^2-1")

ig = ROOT.Math.Integrator()
ig.SetFunction(func)

value = ig.Integral(0, 3)
print("integral-1D value = ", value)
expValue = 6
if (not ROOT.TMath.AreEqualRel(value, expValue, 1.E-15)) :
   print("Error computing integral - computed value - different than expected, diff = ", value - expValue)
   
# example multi-dim function

print("\n\nUse Functor for wrapping a multi-dimensional function, the Rosenbrock Function r(x,y) and find its minimum")

def RosenbrockFunction(xx):
  x = xx[0]
  y = xx[1]
  tmp1 = y-x*x
  tmp2 = 1-x
  return 100*tmp1*tmp1+tmp2*tmp2


func2D = ROOT.Math.Functor(RosenbrockFunction,2)

### minimize multi-dim function using fitter class

fitter = ROOT.Fit.Fitter()
#use a numpy array to pass initial parameter array 
initialParams = np.array([0.,0.], dtype='d')
fitter.FitFCN(func2D, initialParams)
fitter.Result().Print(ROOT.std.cout)
if (not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(0), 1, 1.E-3) or not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(1), 1, 1.E-3)) :
   print("Error minimizing Rosenbrock function ")

## example 1d grad function
## derivative of f(x)= x**2-1

print("\n\nUse GradFunctor1D for making a function object implementing f(x) and f'(x)")

def g(x): return 2 * x

gradFunc = ROOT.Math.GradFunctor1D(f, g)

#check if ROOT has mathmore
prevLevel = ROOT.gErrorIgnoreLevel
ROOT.gErrorIgnoreLevel=ROOT.kFatal
ret = ROOT.gSystem.Load("libMathMore") 
ROOT.gErrorIgnoreLevel=prevLevel
if (ret < 0) :
   print("ROOT has not Mathmore")
   print("derivative value at x = 1", gradFunc.Derivative(1) )

else :
   rf = ROOT.Math.RootFinder(ROOT.Math.RootFinder.kGSL_NEWTON)
   rf.SetFunction(gradFunc, 3)
   rf.Solve()
   value = rf.Root()
   print("Found root value x0 : f(x0) = 0  :  ", value)
   if (value != 1):
      print("Error finding a ROOT of function f(x)=x^2-1")


print("\n\nUse GradFunctor for making a function object implementing f(x,y) and df(x,y)/dx and df(x,y)/dy")

def RosenbrockDerivatives(xx, icoord):
  x = xx[0]
  y = xx[1]
  #derivative w.r.t x 
  if (icoord == 0) :
    return 2*(200*x*x*x-200*x*y+x-1)
  else : 
    return 200 * (y - x * x)
    
gradFunc2d = ROOT.Math.GradFunctor(RosenbrockFunction, RosenbrockDerivatives, 2)

fitter = ROOT.Fit.Fitter()
#here we use a python array to pass initial parameters
initialParams = array.array('d',[0.,0.])
fitter.FitFCN(gradFunc2d, initialParams)
fitter.Result().Print(ROOT.std.cout)
if (not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(0), 1, 1.E-3) or not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(1), 1, 1.E-3)) :
   print("Error minimizing Rosenbrock function ")
