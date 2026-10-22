# \file
# \ingroup tutorial_math
# \notebook -nodraw
#
# Simple script of how to use the TFeldmanCousins class in pyroot.
#
# First, we get a FeldmanCousing calculator by invoking TFeldmanCousins(). 
# This creates an object with default limits: 
# a minimum-signal-value = 0.0 and a maximum-signal-value = 50.0(scaned) 
# Which helps to calculate upto a 90% Confidence-Level(CL).
# Then, prints the results. 
#
# \macro_output
# \macro_code
#
# \author Adrian John Bevan <bevan@SLAC.Stanford.EDU>
# \translator P. P.


import ROOT

TFeldmanCousins = ROOT.TFeldmanCousins

#globals
gROOT = ROOT.gROOT
gSystem = ROOT.gSystem

#void
def FeldmanCousins():
   
   # This checks TFeldmanCousings Class availability. 
   # Raise error if libPhysics wasn't compiled with ROOT.
   if not gROOT.GetClass("TFeldmanCousins"):
      gSystem.Load("libPhysics.so")
  

   global f
   f = TFeldmanCousins() 
   
   # This calculates either the upper or lower limit for 10 observed
   # events with an estimated background of 3.  The calculation of
   # either upper or lower limit will return that limit and fill
   # all data members with both the upper and lower limit for you.
   Nobserved = 10.0
   Nbackground = 3.0
   
   ul = f.CalculateUpperLimit(Nobserved, Nbackground)
   ll = f.GetLowerLimit()
   
   print(f"For { Nobserved} data observed with and estimated background")
   print(f"of {Nbackground} candidates, the Feldman-Cousins method of ")
   print(f"calculating confidence limits gives:")
   print( "\tUpper Limit = " ,  ul )
   print( "\tLower Limit = " ,  ll )
   print( "at the 90% CL")

if __name__ == "__main__" :
   FeldmanCousins()   

