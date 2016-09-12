/// \file
/// \ingroup tutorial_math
/// \notebook
/// Principal Components Analysis (PCA) example
///
/// Example of using TPrincipal as a stand alone class.
///
/// We create n-dimensional data points, where c = trunc(n / 5) + 1
/// are  correlated with the rest n - c randomly distributed variables.
///
/// \macro_output
/// \macro_code
///
/// \authors Rene Brun, Christian Holm Christensen

#include "TPrincipal.h"

void principal(Int_t n=10, Int_t m=10000)
{
   Int_t c = n / 5 + 1;

   cout << "*************************************************" << endl;
   cout << "*         Principal Component Analysis          *" << endl;
   cout << "*                                               *" << endl;
   cout << "*  Number of variables:           " << setw(4) << n
       << "          *" << endl;
   cout << "*  Number of data points:         " << setw(8) << m
       << "      *" << endl;
   cout << "*  Number of dependent variables: " << setw(4) << c
       << "          *" << endl;
   cout << "*                                               *" << endl;
   cout << "*************************************************" << endl;


   // Initilase the TPrincipal object. Use the empty string for the
   // final argument, if you don't wan't the covariance
   // matrix. Normalising the covariance matrix is a good idea if your
   // variables have different orders of magnitude.
   TPrincipal* principal = new TPrincipal(n,"ND");

   // Use a pseudo-random number generator
   TRandom* randumNum = new TRandom;

   // Make the m data-points
   // Make a variable to hold our data
   // Allocate memory for the data point
   Double_t* data = new Double_t[n];
   for (Int_t i = 0; i < m; i++) {

      // First we create the un-correlated, random variables, according
      // to one of three distributions
      for (Int_t j = 0; j < n - c; j++) {
         if (j % 3 == 0)      data[j] = randumNum->Gaus(5,1);
         else if (j % 3 == 1) data[j] = randumNum->Poisson(8);
         else                 data[j] = randumNum->Exp(2);
      }

      // Then we create the correlated variables
      for (Int_t j = 0 ; j < c; j++) {
         data[n - c + j] = 0;
         for (Int_t k = 0; k < n - c - j; k++) data[n - c + j] += data[k];
      }

      // Finally we're ready to add this datapoint to the PCA
      principal->AddRow(data);
   }

   // We delete the data after use, since TPrincipal got it by now.
   delete [] data;

   // Do the actual analysis
   principal->MakePrincipals();

   // Print out the result on
   principal->Print();

   // Test the PCA
   principal->Test();

   // Make some histograms of the orginal, principal, residue, etc data
   principal->MakeHistograms();

   // Make two functions to map between feature and pattern space
   principal->MakeCode();

   // Start a browser, so that we may browse the histograms generated
   // above
   TBrowser* b = new TBrowser("principalBrowser", principal);
}
