// Example on how to use the new Minimizer class in ROOT
//  Show usage with all the possible minimizers. 
// Minimize the Rosenbrock function (a 2D -function)
// This example is described also in 
// http://root.cern.ch/drupal/content/numerical-minimization#multidim_minim
// input : minimizer name + algorithm name
// randomSeed: = <0 : fixed value: 0 random with seed 0; >0 random with given seed 
//
//Author: L. Moneta Dec 2010

#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "TRandom2.h"
#include "TError.h"
#include <iostream>
 
double RosenBrock(const double *xx )
{
  const Double_t x = xx[0];
  const Double_t y = xx[1];
  const Double_t tmp1 = y-x*x;
  const Double_t tmp2 = 1-x;
  return 100*tmp1*tmp1+tmp2*tmp2;
}
 
int NumericalMinimization(const char * minName = "Minuit2",
                          const char *algoName = "" , 
                          int randomSeed = -1)
{
   // create minimizer giving a name and a name (optionally) for the specific
   // algorithm
   // possible choices are: 
   //     minName                  algoName
   // Minuit /Minuit2             Migrad, Simplex,Combined,Scan  (default is Migrad)
   //  Minuit2                     Fumili2
   //  Fumili
   //  GSLMultiMin                ConjugateFR, ConjugatePR, BFGS, 
   //                              BFGS2, SteepestDescent
   //  GSLMultiFit
   //   GSLSimAn
   //   Genetic
   ROOT::Math::Minimizer* min = 
      ROOT::Math::Factory::CreateMinimizer(minName, algoName);

   // set tolerance , etc...
   min->SetMaxFunctionCalls(1000000); // for Minuit/Minuit2 
   min->SetMaxIterations(10000);  // for GSL 
   min->SetTolerance(0.001);
   min->SetPrintLevel(1);

   // create funciton wrapper for minmizer
   // a IMultiGenFunction type 
   ROOT::Math::Functor f(&RosenBrock,2); 
   double step[2] = {0.01,0.01};
   // starting point
    
   double variable[2] = { -1.,1.2};
   if (randomSeed >= 0) { 
      TRandom2 r(randomSeed);
      variable[0] = r.Uniform(-20,20);
      variable[1] = r.Uniform(-20,20);
   }
 
   min->SetFunction(f);
 
   // Set the free variables to be minimized!
   min->SetVariable(0,"x",variable[0], step[0]);
   min->SetVariable(1,"y",variable[1], step[1]);
 
   // do the minimization
   min->Minimize(); 
 
   const double *xs = min->X();
   std::cout << "Minimum: f(" << xs[0] << "," << xs[1] << "): " 
             << min->MinValue()  << std::endl;

   // expected minimum is 0
   if ( min->MinValue()  < 1.E-4  && f(xs) < 1.E-4) 
      std::cout << "Minimizer " << minName << " - " << algoName 
                << "   converged to the right minimum" << std::endl;
   else {
      std::cout << "Minimizer " << minName << " - " << algoName 
                << "   failed to converge !!!" << std::endl;
      Error("NumericalMinimization","fail to converge");
   }
 
   return 0;
}
