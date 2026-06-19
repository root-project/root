# \file
# \ingroup tutorial_math
# \notebook -nodraw
#
# Tutorial illustrating the use of TMath::Binomial function.
# It can be run with:
#
# ~~~{.py}
# IP[0]: %run binomial.py
# ~~~
#
# \macro_output
# \macro_code
#
# \author Federico Carminati
# \translator P. P.

import ROOT

TMath = 		 ROOT.TMath
TRandom = 		 ROOT.TRandom

#types
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t

#utils
def printf(string, *args):
   print( string%args , end = "")

def to_c(ls):
   return (c_double * len(ls) ) ( * ls)

def to_py(c_ls):
   return list( c_ls )


#void
def binomialSimple():
   #
   # Simple test for the binomial distribution
   #
   printf("\nTMath.Binomial simple test\n")
   printf("Build the Tartaglia triangle\n")
   printf("============================\n")
   Max = Int_t(13)
   j = Int_t()
   #for(Int_t i=0; i<max;i++)
   for i in range(0, Max, 1):
      printf("n=%2d",i);
      #for(j=1;j<(max-i);j++) 
      for j in range(1, Max-i, 1): printf("  ")
      #for(j=0;j<i+1;j++) 
      for j in range(0, i+1, 1):   printf("%4d",TMath.Nint(TMath.Binomial(i,j)))
      printf("\n");
   

#void
def binomialFancy():

   #Initializing.
   x = Double_t()
   y = Double_t()
   res1 = Double_t()
   res2 = Double_t()
   err = Double_t()
   serr = Double_t(0)

   nmax = Int_t(10000)

   printf("\nTMath.Binomial fancy test\n")
   printf("Verify Newton formula for (x+y)^n\n")
   printf("x,y in [-2,2] and n from 0 to 9  \n")
   printf("=================================\n")
   
   global r
   r = TRandom()
   #for(Int_t i=0 ;i<nmax ;i++)
   for i in range(0, nmax, 1):
      #Adef large cancellations
      while ( TMath.Abs( x + y ) < 0.75 ) : 
         x=2 * (1 -2 * r.Rndm())
         y=2 * (1 -2 * r.Rndm())

      #for(Int_t j=0 ;j<10 ;j++)
      for j in range(0, 10, 1):

         res1 = TMath.Power(x+y,j)
         res2 = 0

         #for(Int_t k=0;k<=j;k++)
         k = 0
         while( k <= j ): 
            res2 += TMath.Power(x,k) * TMath.Power(y,j-k) * TMath.Binomial(j,k)
            k += 1

         err = TMath.Abs(res1-res2)/TMath.Abs(res1) 
         if( err > 1e-10 ):
            printf("res1=%e res2=%e x=%e y=%e err=%e j=%d\n",res1,res2,x,y,err,j)

         serr += err

         
      
   printf("Average Error = %e\n", serr/nmax);
   #printf(f"Average Error = {serr / nmax } \n")

   

#void
def binomial():
   binomialSimple()
   binomialFancy()
   

if __name__ == "__main__":
   binomial() 
 

