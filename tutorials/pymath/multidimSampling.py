## \file
## \ingroup tutorial_math
## \notebook
## 
## Example on how to sample a multi-dimensional distribution 
## by using the DistSampler class.
## NOTE: 
##       This tutorial must be run with ACLIC
##
## \macro_image
## \macro_code
##
## \author Lorenzo Moneta
## \translator P. P.


import ROOT
import ctypes
from struct import Struct


TMath = ROOT.TMath 
TF1 = ROOT.TF1
TF2 = ROOT.TF2 
TStopwatch = ROOT.TStopwatch 
TKDTreeBinning = ROOT.TKDTreeBinning 
TTree = ROOT.TTree 
TFile = ROOT.TFile 
TMatrixDSym = ROOT.TMatrixDSym 
TVectorD = ROOT.TVectorD 
TCanvas = ROOT.TCanvas 
TString = ROOT.TString

#cmath = ROOT.cmath  # Note implemented

#standar library
std = ROOT.std
sqrt = std.sqrt
Error = ROOT.Error



#Math
Math = ROOT.Math
DistSampler = Math.DistSampler 
DistSamplerOptions = Math.DistSamplerOptions 
MinimizerOptions = Math.MinimizerOptions 
Factory = Math.Factory 

#types
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t
c_double = ctypes.c_double
nullptr = ROOT.nullptr

#cpp integration
ProcessLine = ROOT.gInterpreter.ProcessLine

#utils
def to_c(ls):
   return ( c_double * len(ls) )( *ls )

#flags 
debug = False

# function (a 4d gaussian)



#Note: 
#     Gauss ND is a function that
#     makes(creates) a class in order to avoid the construction of 
#     matrices for every call in a lon run.
#     In .C :
#            This, however, requires that the code must be compiled with ACLIC
#     In .py :
#            No need for compilation with ACLIC. Just have instaleld pyroot version > 6.30.

# Define GausND structure
# struct
class GausND(Struct) :
   
   X = TVectorD()
   Mu = TVectorD()
   CovMat = TMatrixDSym()
   
   def __init__(self, dim = Int_t(0)) :
      self.X      = TVectorD(dim)
      self.Mu     = TVectorD(dim)
      self.CovMat = TMatrixDSym(dim)
   
   
   def __call__(self, x : Double_t, p : Double_t) :
      # 4 parameters
      dim = self.X.GetNrows()
      k = 0

      #for (i = 0; i<dim; ++i)  X[i] = x[i] - p[k]; k++;
      for i in range(0, dim, 1):
         self.X[i] = x[i] - p[k]
         k += 1

      #for (int i = 0; i<dim; ++i) {
      for i in range(0, dim, 1):
         self.CovMat[i,i] = p[k] * p[k]
         k += 1
         
      #for (int i = 0; i<dim; ++i) {
      #   for (int j = i+1; j<dim; ++j) {
      for i in range(0, dim, 1):
         for j in range(i+1, dim, 1):
            # p now are the correlations N(N-1)/2
            self.CovMat[i,j] = p[k] * sqrt( self.CovMat[i,i] * self.CovMat[j,j] )
            self.CovMat[j,i] = self.CovMat[i,j]
            k += 1
            
         
      if debug:
         self.X.Print()
         self.CovMat.Print()
         
      
      det = self.CovMat.Determinant()
      if (det <= 0):
         Fatal("GausND","Determinant is <= 0 det = %f", det)
         self.CovMat.Print()
         return 0
         
      norm = std.pow( 2. * TMath.Pi(), dim/2) * sqrt(det)
      # compute the gaussians
      self.CovMat.Invert()
      fval = std.exp( - 0.5 * self.CovMat.Similarity(self.X) )/ norm
      
      if debug:
         print(f"det  " , det)
         print(f"norm " , norm)
         print(f"fval " , fval)
         
      
      return fval
      
   

# Use the Math namespace

# void
def multidimSampling() :
   
   
   N = 10000
   #NBin = 1000
   DIM = 4
   
   xmin = [ -10,-10,-10, -10 ]
   xmax = [  10, 10, 10, 10 ]
   par0 = [  1., -1., 2, 0 # \mu of the gaussian-distribution
           , 1, 2, 1, 3 # \sigma of the same gaussian-distribution
           , 0.5,0.,0.,0.,0.,0.8 # its correlation
             ]
   
   NPAR = DIM + DIM*(DIM+1)//2; # 14 in the 4-dim case.

   # generate the sample
   global gaus4d, f, param_functor
   gaus4d = GausND(4)
   param_functor = Math.ParamFunctor(gaus4d.__call__)
   #f = TF1("functionND", gaus4d, 0, 1, 14)
   f = TF1(name = "GausND", f=param_functor, xmin = 0, xmax = 1, npar =14)
   c_par0 = to_c( par0 )
   f.SetParameters(c_par0)
   #Note: 
   #      TF1 invokes the next constructor.
   """
   TF1::TF1(const char* name, ROOT::Math::ParamFunctor f, Double_t xmin = 0, Double_t xmax = 1, Int_t npar = 0, Int_t ndim = 1, TF1::EAddToList addToGlobList = EAddToList::kDefault)
   """
   
   x0 = [ 0,0,0,0 ]

   # for debugging
   global debug
   if (debug):
      f.EvalPar(x0,nullptr)
   debug = False
   
   global name
   name = TString()
   #for (int i = 0; i < NPAR; ++i )  {
   for i in range(0, NPAR, 1):
      if (i < DIM):
          f.SetParName(i, str(name.Format("mu_%d"%(i+1)) ) )
      elif (i < 2 * DIM):
          f.SetParName(i, str(name.Format("sig_%d"%(i-DIM+1)) ) ) 
      elif (i < 2 * DIM):
          f.SetParName(i, str(name.Format("sig_%d"%(i-2*DIM+1)) ) )
      
   
   '''ROOT.Math.DistSamplerOptions.SetDefaultSampler("Foam")'''
   #Not to use: 
   #sampler = Factory.CreateDistSampler() # -> TUnuranSampler 
   #It should be a DistSampler-object. 
   #Instead:
   ProcessLine("""
   ROOT::Math::DistSampler * sampler = ROOT::Math::Factory::CreateDistSampler();
   """) 
   global sampler
   sampler = ROOT.sampler
   

   if (sampler == nullptr):
      Info("multidimSampling",
           "Default sampler %s is not available try with Foam "%ROOT.Math.DistSamplerOptions.DefaultSampler()
           )
      ROOT.Math.DistSamplerOptions.SetDefaultSampler("Foam")
      
   #sampler = Factory.CreateDistSampler()
   ProcessLine("""
   ROOT::Math::DistSampler * sampler = ROOT::Math::Factory::CreateDistSampler();
   """) 
   sampler = ROOT.sampler
   if (sampler == nullptr):
      Error("multidimSampling","Foam sampler is not available - exit ")
      return
      
   
   #BP:
   #Not to use: 
   #sampler.SetFunction(f,DIM)
   #sampler.SetFunction["TF1"](func = f, dim = DIM)
   #Instead:
   sampler.SetFunction(f,DIM)
   #This works too. Just in case you'll need to set a differnt kind of function.
   #sampler.SetFunction["TF1"](f,DIM)
   #Using Math.DistSampler as an abstract class does't function well.
   #class DistSamplerPy(Math.DistSampler):
   #   def __init__(self):
   #      super(Math.DistSampler, self).__init__()
   #global samplerPy
   #samplerPy = DistSamplerPy() 
   # Because f is a TF1-object, we use "TF1" as the template parameter.
   #samplerPy.SetFunction["TF1"]( f, DIM)    
   # it gives error. None of the overloaded template functions correspond.
   #Continuing ...
   sampler.SetRange(xmin,xmax)

   global ret
   ret = sampler.Init()
   
   global data1, v, w
   data1 = std.vector["Double_t"](DIM * N)
   v = [ Double_t() for _ in range(DIM) ] 
   #Convertion to C-types.
   v = to_c(v)
   w = TStopwatch()
   
   if not ret:
      Error("Sampler::Init","Error initializing unuran sampler")
      return
      
   
   # generate the data
   w.Start()
   #for (int i = 0; i < N; ++i) {
   #   for( j = 0; j < DIM; ++j)
   for i in range(0, N, 1):
      sampler.Sample(v)
      for j in range(0, DIM, 1):
         data1[ N*j + i ] = v[j];
      
   w.Stop()
   w.Print()
   
   # fill the tree with data
   global file
   file = TFile("multiDimSampling.root","RECREATE")
   x = [ Double_t() for _ in range(DIM) ]
   x = to_c(x)
   global t1
   t1 = TTree("t1","Tree from Unuran")
   t1.Branch("x",x,"x[4]/D")

   #for (int i = 0; i < N; ++i) {
   #   for (int j = 0; j < DIM; ++j) {
   for i in range(0, N, 1):
      for j in range(0, DIM, 1):
         x[j] = data1[i+N*j]
      t1.Fill()
      
   
   # plot the data
   t1.Draw("x[0]:x[1]:x[2]:x[3]","","candle")

   global c2
   c2 = TCanvas()
   c2.Divide(3,2)

   ic = Int_t(0)
   c2.cd( (ic:=ic+1) )
   t1.Draw("x[0]:x[1]")
   c2.cd( (ic:=ic+1) )
   t1.Draw("x[0]:x[2]")
   c2.cd( (ic:=ic+1) )
   t1.Draw("x[0]:x[3]")
   c2.cd( (ic:=ic+1) )
   t1.Draw("x[1]:x[2]")
   c2.cd( (ic:=ic+1) )
   t1.Draw("x[1]:x[3]")
   c2.cd( (ic:=ic+1) )
   t1.Draw("x[2]:x[3]")
   
   t1.Write()
   file.Close()
   
   


if __name__ == "__main__":
   multidimSampling()
