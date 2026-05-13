## \file
## \ingroup tutorial_math
## \notebook
## Tutorial illustrating how to create a TF1 class using 
## a functor or a class as member functions.
##
## This script can be run with:
##
## ~~~{.py}
##  IP[0] %run exampleFunctor.py
## ~~~
##
## \macro_image
## \macro_code
##
## \author Lorenzo Moneta
## \translator P. P.


import ROOT
import ctypes
from struct import Struct


TF1 = ROOT.TF1 
TMath = ROOT.TMath 
TLegend = ROOT.TLegend 

#math
Math = ROOT.Math
Functor = Math.Functor
Functor1D = Math.Functor1D

#c_types
#ctypes = cppyy.ctypes
c_double = ctypes.c_double

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kMagenta = ROOT.kMagenta

#utils
# convertion to c_array
def to_c(ls):
   return (c_double * len(ls))(*ls)

def to_py(c_ls):
   return [ _ for _ in c_ls]


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# Double_t 
def MyFunc (x : Double_t , p : Double_t ) :
   return TMath.Gaus(x[0],p[0],p[1] )
   

# function object (functor)
# struct
class MyDerivFunc(Struct) :
#class MyDerivFunc :
   
   def __init__(self, f : TF1 = TF1() ):
      self.__fFunc__ = f

   # Your favorite operation.
   # Let's say ... .Derivative(x)
   def __call__(self, x : Double_t, p: Double_t=None ) : 
      #In case you want to use ROOT.Math.Functor.
      #def __call__(self, x : list[Double_t], p: list[Double_t]) : 
      #   return self.__fFunc__.Derivative( x[0] ) #  
   
      #Note:
      #      It is important to notice x[0] as an argument instead of just x.
      #      The reason behind this scene relies on how ROOT.Math.Functor is 
      #      implemented. Try not to execute in Python MyDerivFunc.__call__(int or List).
      #      It'll give you error and one value always, respectively
      #      However, we will use lambdas in this case as it is no longer needed.
      #      See down there, when we are defining f1,f2,f3, how is it implemented.

      return self.__fFunc__.Derivative( x ) #  
      
   __fFunc__ = TF1()
   
# function class with a member function
# struct
#class MyIntegFunc(Struct) :
class MyIntegFunc(Struct):

   def __init__(self, f : TF1 = TF1() ):
      #super(Struct, self).__init__()
      self.__fFunc__ = f
   #Double_t
   #def Integral(self, x : list[Double_t], empty_slot: Double_t =None) :
   def Integral(self, x : Double_t, p: Double_t = None) :
      # paramaters p
      self.param1 = p
      self.param2 = p 
      # or like *args
      #param1, param2 = args
      
      # operation a
      a = self.__fFunc__.GetXmin()
      return self.__fFunc__.Integral( a, x )

   def __call__(self, x):
      # Through self, we access the parameters.
      # param1 = self.param1
      # param2 = self.param2 
      return Integral( x)
      
   __fFunc__ = TF1()
   



# void
def exampleFunctor() :
   global xmin, xmax
   
   xmin = -10
   xmax = 10
   
   # create TF1 using a free C function
   global f1
   f1 = TF1("f1",MyFunc,xmin,xmax,2)
   f1.SetParameters(0.,1.)
   f1.SetLineColor( kRed )
   f1.SetMaximum(3)
   f1.SetMinimum(-1)
   f1.Draw()
   
   # The Derivative function.
   # Example on how to create a TF1-object using a functor.
   
   # In order to work with the interpreter, the function object must be 
   # created and should live "all time for all time".
   # Of the TF1-object, in compiled mode, the function object can be passed 
   # by value(strongly recommended), and also there is
   # no need to specify the type of the function-class. 
   # For example:
   #
   # In .C version:
   # `TF1 * f2 = new TF1("f2",MyDerivFunc(f1), xmin, xmax,0); # only for C++ compiled mode`
   # In .py version:
   # `deriv_lambda = lambda x, p:  MyDerivFunc(f1).__call__(x[0]) `
   # `f2 = TF1("f2", deriv_lambda, xmin, xmax,0) ` 
   
   global deriv, f2
   #DOING
   #BP: At using ROOT.Math.Functor while constructiing TF1(... Functor ...). 
   deriv = MyDerivFunc(f1)

   #No longer needed.
   #global deriv_Functor
   #deriv_Functor = Functor(f = deriv.__call__, dim = 10) # 1 variable, 1 parameter
   
   global deriv_lambda
   deriv_lambda = lambda x, p : deriv.__call__(x[0])  
  
   f2 = TF1( "f2",  deriv_lambda,  xmin, xmax, 0, 1) # 0=parm, 1=ndim
   f2.SetLineColor(kBlue)
   f2.Draw("same")
   
   
   # The Integral Function.
   # An example on how to create a TF1-object using a member-function of
   # a user-defined-class.
   
   # Same as before.
   # In order to work with the interpreter, the function object must be 
   # created and should live "for ever."
   # Here change a little. Read carefully!
   # Of the TF1, in compiled mode, there is no need to specify the type 
   # of the function-class and the name of the member-function.
   #
   # In .C version:
   # `TF1 * f3 = new TF1("f3",intg,&MyIntegFunc::Integral,xmin,xmax, 0); # only for C++ compiled mode`
   # In .py version:
   # intg_Integral_lambda = lambda x, p: intg.Integral(x[0])
   # `f3 = TF1("f3", intg, MyIntegFunc.Integral, xmin, xmax, 0) ` 
   
   global intg, f3
   #intg = MyIntegFunc(f1)
   #Note :
   #      We use intg.__call__ as an argument when we use TF1. 
   #      If we use intg alone, raises error.
   #f3 = TF1("f3", intg, MyIntegFunc.Integral, xmin, xmax, 0)
   #f3 = TF1("f3", intg.Integral, xmin, xmax, 0)
   #BP:
   ##global intg_Integral_Functor
   ##intg_Integral_Functor = Functor(f = intg.Integral, dim = 1) 
   intg = MyIntegFunc(f1)

   global intg_Integral_lambda
   intg_Integral_lambda = lambda x, p: intg.Integral(x[0])

   f3 = TF1("f3", intg_Integral_lambda, xmin, xmax, 0, 1) #par 
   
   f3.SetLineColor( kMagenta )
   f3.Draw("same")
   
   global l
   l = TLegend(0.78, 0.25, 0.97,0.45)
   l.AddEntry(f1, "Func")
   l.AddEntry(f2, "Deriv.")
   l.AddEntry(f3, "Integral")
   l.Draw()
   


if __name__ == "__main__":
   exampleFunctor()
