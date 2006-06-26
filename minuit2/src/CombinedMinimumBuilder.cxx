// @(#)root/minuit2:$Name:  $:$Id: CombinedMinimumBuilder.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/CombinedMinimumBuilder.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {

   namespace Minuit2 {


FunctionMinimum CombinedMinimumBuilder::Minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double edmval) const {
   // find minimum using combined method 
   // (Migrad then if fails try Simplex and then Migrad again) 
   
   FunctionMinimum min = fVMMinimizer.Minimize(fcn, gc, seed, strategy, maxfcn, edmval);
   
   if(!min.IsValid()) {
      std::cout<<"CombinedMinimumBuilder: migrad method fails, will try with simplex method first."<<std::endl; 
      MnStrategy str(2);
      FunctionMinimum min1 = fSimplexMinimizer.Minimize(fcn, gc, seed, str, maxfcn, edmval);
      if(!min1.IsValid()) {
         std::cout<<"CombinedMinimumBuilder: both migrad and simplex method fail."<<std::endl;
         return min1;
      }
      MinimumSeed seed1 = fVMMinimizer.SeedGenerator()(fcn, gc, min1.UserState(), str);
      
      FunctionMinimum min2 = fVMMinimizer.Minimize(fcn, gc, seed1, str, maxfcn, edmval);
      if(!min2.IsValid()) {
         std::cout<<"CombinedMinimumBuilder: both migrad and method fails also at 2nd attempt."<<std::endl;
         std::cout<<"CombinedMinimumBuilder: return simplex Minimum."<<std::endl;
         return min1;
      }
      
      return min2;
   }
   
   return min;
}

   }  // namespace Minuit2

}  // namespace ROOT
