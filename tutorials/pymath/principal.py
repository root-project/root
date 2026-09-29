## \file
## \ingroup tutorial_math
## \notebook
##
## The Principal Components Analysis (PCA) example.
##
## Example of how to PCA as a stand alone class.
##
## We create n-dimensional data points, where c = trunc(n / 5) + 1
## are correlated with the rest n - c randomly distributed variables.
##
## \macro_output
## \macro_code
##
## \authors Rene Brun, Christian Holm Christensen
## \translator P. P.


import ROOT
import cppyy

#import ctypes # Not to use. ctypes can or cannot be in their system.
ctypes = cppyy.ctypes # cppyy alwyas will have ctypes as an inner module.

TPrincipal = ROOT.TPrincipal 
TRandom = ROOT.TRandom
TBrowser = ROOT.TBrowser

#standar library
std = ROOT.std
setw = std.setw 

#types
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t
c_double = ctypes.c_double

#utils 
def to_c(ls):
   return (c_double * len(ls) )( * ls )
#
Remove = ROOT.gROOT.Remove




# void
def principal(n : Int_t = 10, m : Int_t = 10000) :
   c = n // 5 + 1
   
      
   print(f"*************************************************")
   print(f"*         Principal Component Analysis          *")
   print(f"*                                               *")
   print(f"*  Number of variables:           " , setw(4) , n, "          *")
   print(f"*  Number of data points:         " , setw(8) , m, "          *")
   print(f"*  Number of dependent variables: " , setw(4) , c, "          *")
   print(f"*                                               *")
   print(f"*************************************************")
   
   
   # Initilase a TPrincipal object. Use an empty string for the
   # final argument if you don't want the covariance
   # matrix. Normalizing(or normalising) the covariance matrix is a good idea if your
   # variables have different orders of magnitude.
   global principal
   principal = TPrincipal(n,"ND")
   
   # Use a pseudo-random number generator
   global randumNum
   randumNum = TRandom()
   
   # Make the m data-points
   # Make a variable to hold our data
   # Allocate memory for the data point
   global data, c_data
   data = [ Double_t() for _ in range(n) ]  
   #for (Int_t i = 0; i < m; i++) {
   for i in range(0, m, 1):
   
      
      # First we create the un-correlated, random variables, according
      # to one of three distributions
      #for (Int_t j = 0; j < n - c; j++) {
      #for j in range(0, n - c, 1):
      j = 0
      while( j < n - c ):
         if (j % 3 == 0):      data[j] = randumNum.Gaus(5,1)
         elif (j % 3 == 1):    data[j] = randumNum.Poisson(8)
         else :                data[j] = randumNum.Exp(2)
         j += 1
         
      
      # Then we create the correlated variables
      #for (Int_t j = 0 ; j < c; j++) {
      for j in range(0, c, 1):
         data[n - c + j] = 0;
         #for (Int_t k = 0; k < n - c - j; k++) 
         k = 0 
         while( k < n - c - j ):
            data[n - c + j] += data[k]
            k += 1
         
      
      # Finally we're ready to add this datapoint to the PCA
      c_data = to_c( data )
      principal.AddRow( c_data )
      data = list(data)
      
   
   # We delete the data after use, since the TPrincipal-object got it by now.
   # Remove(data) # raises error. 
   # data.Clear() # data is not a TObject.
   #for _i in range(len(data)):
   #   _i.Clear() 
   #   Remove(_i)
   #   del _i
   #del data
   
   # Do the actual analysis.
   principal.MakePrincipals()
   
   # Print out the result on.
   principal.Print()
   
   # Test the PCA.
   principal.Test()
   
   # Make some histograms of the original, principal, residuals, etc. data.
   principal.MakeHistograms()
   
   # Make two functions to map between feature and pattern space
   principal.MakeCode()
   
   # Start a browser, so that we may browse the histograms generated
   # above
   global b
   b = TBrowser("principalBrowser", principal)
   
   #To avoid potential memory leak.
   #ROOT.gROOT.DeleteAll()
   # However after > 3 runs it still generates memory issues in those TH1 
   # histograms created by TPrincipal.


if __name__ == "__main__":
   principal()
