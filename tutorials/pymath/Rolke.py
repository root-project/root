## \file
## \ingroup tutorial_math
## \notebook -nodraw
## Example of the usage of the TRolke class
## The TRolke class computes the profile likelihood
## confidence limits for 7 different model assumptions
## on systematic/statistical uncertainties
##
## Please read TRolke.cxx and TRolke.h for more docs.
##
## \macro_output
## \macro_code
##
## \authors Jan Conrad, Johan Lundberg
## \translator P. P.


import ROOT
import ctypes

TROOT = ROOT.TROOT 
TSystem = ROOT.TSystem 
TRolke = ROOT.TRolke 
#Riostream = ROOT.Riostream 

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double
c_int = ctypes.c_int

# void
def Rolke() :
   # variables used throughout the example
   bm = 	 Double_t()
   tau = 	 Double_t()
   mid = 	 Int_t()
   m = 	 Int_t()
   z = 	 Int_t()
   y = 	 Int_t()
   x = 	 Int_t()
   e = 	 Double_t()
   em = 	 Double_t()
   sde = 	 Double_t()
   sdb = 	 Double_t()
   b = 	 Double_t()
   
   alpha = Double_t() #Confidence Level
   
   # make TRolke objects
   tr = TRolke()   #
   
   ul = c_double() # upper limit
   ll = c_double() # lower limit
   
   #-----------------------------------------------
   # Model 1 assumes:
   #
   # Poisson uncertainty in the background estimate
   # Binomial uncertainty in the efficiency estimate
   #
   print("\n ======================================================== ") 
   mid =1
   x = 5; # events in the signal region
   y = 10; # events observed in the background region
   tau = 2.5; # ratio between size of signal/background region
   m = 100; # MC events have been produced (signal)
   z = 50; # MC events have been observed (signal)
   
   alpha=0.9; #Confidence Level
   
   tr.SetCL(alpha)
   
   tr.SetPoissonBkgBinomEff(x,y,z,tau,m)
   tr.GetLimits(ll,ul)
   
   print(f"For model 1: Poisson / Binomial")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   #-----------------------------------------------
   # Model 2 assumes:
   #
   # Poisson uncertainty in the background estimate
   # Gaussian  uncertainty in the efficiency estimate
   #
   print("\n ======================================================== ") 
   mid =2
   y = 3 ; # events observed in the background region
   x = 10 ; # events in the signal region
   tau = 2.5; # ratio between size of signal/background region
   em = 0.9; # measured efficiency
   sde = 0.05; # standard deviation of efficiency
   alpha =0.95; # Confidence Level
   
   tr.SetCL(alpha)
   
   tr.SetPoissonBkgGaussEff(x,y,em,tau,sde)
   tr.GetLimits(ll,ul)
   
   print(f"For model 2 : Poisson / Gaussian")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   #-----------------------------------------------
   # Model 3 assumes:
   #
   # Gaussian uncertainty in the background estimate
   # Gaussian  uncertainty in the efficiency estimate
   #
   print("\n ======================================================== ") 
   mid =3
   bm = 5; # expected background
   x = 10; # events in the signal region
   sdb = 0.5; # standard deviation in background estimate
   em = 0.9; # measured efficiency
   sde = 0.05; # standard deviation of efficiency
   alpha =0.99; # Confidence Level
   
   tr.SetCL(alpha)
   
   tr.SetGaussBkgGaussEff(x,bm,em,sde,sdb)
   tr.GetLimits(ll,ul)

   print(f"For model 3 : Gaussian / Gaussian")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   print(f"***************************************")
   print(f"* some more examples for gauss/gauss *")
   print(f"*                                     *")
   slow = c_double()
   shigh = c_double()
   tr.GetSensitivity(slow,shigh)
   print(f"sensitivity:")
   print(f"[" , round(slow.value, 5) , "," , round(shigh.value, 5) , "]")
   
   outx = c_int()
   tr.GetLimitsQuantile(slow,shigh,outx,0.5)
   print(f"median limit:")
   print(f"[" , round(slow.value, 5) , "," , round(shigh.value, 5) , "] @ x =" , outx.value  )
   
   tr.GetLimitsML(slow,shigh,outx)
   print(f"ML limit:")
   print(f"[" , round(slow.value, 5) , "," , round(shigh.value, 5) , "] @ x =" , outx.value  )
   
   tr.GetSensitivity(slow,shigh)
   print(f"sensitivity:")
   print(f"[" , round(slow.value, 5) , "," , round(shigh.value, 5) , "]")
   
   ll = c_double()
   ul = c_double()
   tr.GetLimits(ll,ul)
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   ncrt = c_int()
   
   tr.GetCriticalNumber(ncrt)
   print(f"critical number: " , ncrt.value)
   
   tr.SetCLSigmas(5)
   tr.GetCriticalNumber(ncrt)
   print(f"critical number for 5 sigma: " , ncrt.value)
   
   print(f"***************************************")
   
   #-----------------------------------------------
   # Model 4 assumes:
   #
   # Poisson uncertainty in the background estimate
   # known efficiency
   #
   print("\n ======================================================== ") 
   mid =4
   y = 7; # events observed in the background region
   x = 1; # events in the signal region
   tau = 5; # ratio between size of signal/background region
   e = 0.25; # efficiency
   
   alpha =0.68; # Confidence Level
   
   tr.SetCL(alpha)
   
   tr.SetPoissonBkgKnownEff(x,y,tau,e)
   tr.GetLimits(ll,ul)
   
   print(f"For model 4 : Poissonian / Known")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   #-----------------------------------------------
   # Model 5 assumes:
   #
   # Gaussian uncertainty in the background estimate
   # Known efficiency
   #
   print("\n ======================================================== ") 
   mid =5
   bm = 0; # measured background expectation
   x = 1 ; # events in the signal region
   e = 0.65; # known eff
   sdb = 1.0; # standard deviation of background estimate
   alpha =0.799999; # Confidence Level
   
   tr.SetCL(alpha)
   
   tr.SetGaussBkgKnownEff(x,bm,sdb,e)
   tr.GetLimits(ll,ul)
   
   print(f"For model 5 : Gaussian / Known")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   #-----------------------------------------------
   # Model 6 assumes:
   #
   # Known background
   # Binomial uncertainty in the efficiency estimate
   #
   print("\n ======================================================== ") 
   mid =6
   b = 10; # known background
   x = 25; # events in the signal region
   z = 500; # Number of observed signal MC events
   m = 750; # Number of produced MC signal events
   alpha =0.9; # Confidence L evel
   
   tr.SetCL(alpha)
   
   tr.SetKnownBkgBinomEff(x, z,m,b)
   tr.GetLimits(ll,ul)
   
   print(f"For model 6 : Known / Binomial")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   #-----------------------------------------------
   # Model 7 assumes:
   #
   # Known Background
   # Gaussian  uncertainty in the efficiency estimate
   #
   print("\n ======================================================== ") 
   mid =7
   x = 15; # events in the signal region
   em = 0.77; # measured efficiency
   sde = 0.15; # standard deviation of efficiency estimate
   b = 10; # known background
   alpha =0.95; # Confidence L evel
   
   y = 1
   
   tr.SetCL(alpha)
   
   tr.SetKnownBkgGaussEff(x,em,sde,b)
   tr.GetLimits(ll,ul)
   
   print(f"For model 7 : Known / Gaussian ")
   print(f"the Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   #-----------------------------------------------
   # Example of bounded and unbounded likelihood
   # Example for Model 1
   
   bm = 0.0
   tau = 5
   mid = 1
   m = 100
   z = 90
   y = 15
   x = 0
   alpha = 0.90
   
   tr.SetCL(alpha)
   tr.SetPoissonBkgBinomEff(x,y,z,tau,m)
   tr.SetBounding(True); #bounded
   tr.GetLimits(ll,ul)
   
   print(f"Example of the effect of bounded vs unbounded, For model 1")
   print(f"the BOUNDED Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   
   
   tr.SetBounding(False); #unbounded
   tr.GetLimits(ll,ul)
   
   print(f"the UNBOUNDED Profile Likelihood interval is :")
   print(f"[" , round(ll.value, 5) , "," , round(ul.value, 5) , "]")
   


if __name__ == "__main__":
   Rolke()
