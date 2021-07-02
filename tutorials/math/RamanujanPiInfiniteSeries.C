/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Tutorial demonstrating how infinite series can be used with
/// S. Ramanujan's Formula for calculating Pi.
///
/// \macro_output
/// \macro_code
///
/// \author Advait Dhingra

#include <TMath.h>
#include <iostream>

void RamanujanPiInfiniteSeries() {
   Int_t iterations;

   std::cout << "Number of Iterations: " << std::endl;
   std::cin >> iterations;

   Double_t summation;

   for (int n = 0; n < iterations; n++) {
      summation += (TMath::Factorial(4*n) / std::pow(TMath::Factorial(n), 4)) * ((26390*n + 1103) / std::pow(396, 4*n));
   }

   Double_t pi = 1/((TMath::Sqrt(8) / 9801) * summation);

   std::cout << std::setprecision(iterations) << pi << std::endl;

}
