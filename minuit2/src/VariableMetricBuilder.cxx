// @(#)root/minuit2:$Name:  $:$Id: VariableMetricBuilder.cxx,v 1.8 2006/10/27 11:21:14 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/VariableMetricBuilder.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnPosDef.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/LaSum.h"
#include "Minuit2/LaProd.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnPrint.h"

//#define DEBUG 


#ifdef DEBUG
#define WARNINGMSG
#include "Minuit2/MnPrint.h"
#endif

namespace ROOT {

   namespace Minuit2 {


double inner_product(const LAVector&, const LAVector&);

FunctionMinimum VariableMetricBuilder::Minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double edmval) const {   
   // top level function to find minimum from a given initial seed 
   // iterate on a minimum search in case of first attempt is not succesfull
   
   edmval *= 0.0001;
   
   
#ifdef DEBUG
   std::cout<<"VariableMetricBuilder convergence when edm < "<<edmval<<std::endl;
#endif
   
   if(seed.Parameters().Vec().size() == 0) {
      return FunctionMinimum(seed, fcn.Up());
   }
   
   
   //   double edm = Estimator().Estimate(seed.Gradient(), seed.Error());
   double edm = seed.State().Edm();
   
   FunctionMinimum min(seed, fcn.Up() );
   
   if(edm < 0.) {
#ifdef WARNINGMSG
      std::cout<<"VariableMetricBuilder: initial matrix not pos.def."<<std::endl;
#endif
      //assert(!seed.Error().IsPosDef());
      return min;
   }
   
   std::vector<MinimumState> result;
   //   result.reserve(1);
   result.reserve(8);
   
   result.push_back( seed.State() );
   
   // do actual iterations
   
   
   // try first with a maxfxn = 80% of maxfcn 
   int maxfcn_eff = maxfcn;
   int ipass = 0;
   
   do { 
      
#ifdef DEBUG
      std::cout << "start iterating... " << std::endl; 
      if (ipass > 0)  std::cout << "continue iterating... " << std::endl; 
#endif
      
      min = Minimum(fcn, gc, seed, result, maxfcn_eff, edmval);
      // second time check for validity of function Minimum 
      if (ipass > 0) { 
         if(!min.IsValid()) {
#ifdef WARNINGMSG
            std::cout<<"FunctionMinimum is invalid."<<std::endl;
#endif
            return min;
         }
      }
      
      // resulting edm of minimization
      edm = result.back().Edm();
      
      if( (strategy.Strategy() == 2) || 
          (strategy.Strategy() == 1 && min.Error().Dcovar() > 0.05) ) {
         
#ifdef DEBUG
         std::cout<<"MnMigrad will verify convergence and Error matrix. "<< std::endl;
         std::cout<<"dcov is =  "<<  min.Error().Dcovar() << std::endl;
#endif
         
         MinimumState st = MnHesse(strategy)(fcn, min.State(), min.Seed().Trafo());
         result.push_back( st );
         
         // check edm 
         edm = st.Edm();
#ifdef DEBUG
         std::cout << "edm after Hesse calculation " << edm << " requested " << edmval << std::endl;
#endif
         if (edm > edmval) { 
#ifdef WARNINGMSG
            std::cout << "VariableMetricBuilder: Tolerance is not sufficient - edm is " << edm << " requested " << edmval 
            << " continue the minimization" << std::endl;
#endif
         }
      }
      
      
      // end loop on iterations
      // ? need a maximum here (or max of function calls is enough ? ) 
      // continnue iteration (re-calculate funciton Minimum if edm IS NOT sufficient) 
      // no need to check that hesse calculation is done (if isnot done edm is OK anyway)
      // count the pass to exit second time when function Minimum is invalid
      // increase by 20% maxfcn for doing some more tests
      if (ipass == 0) maxfcn_eff = int(maxfcn*1.3);
      ipass++;
   }  while (edm > edmval );
   
   
   
   // Add latest state (Hessian calculation)
   min.Add( result.back() );
   
   return min;
}

FunctionMinimum VariableMetricBuilder::Minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, std::vector<MinimumState>& result, unsigned int maxfcn, double edmval) const {
   // function performing the minimum searches using the Variable Metric  algorithm (MIGRAD) 
   // perform first a line search in the - Vg direction and then update using the Davidon formula (Davidon Error updator)
   // stop when edm reached is less than required (edmval)
   
   // after the modification when I iterate on this functions, so it can be called many times, 
   //  the seed is used here only to get precision and construct the returned FunctionMinimum object 
   

   
   const MnMachinePrecision& prec = seed.Precision();
   
   
   //   result.push_back(MinimumState(seed.Parameters(), seed.Error(), seed.Gradient(), edm, fcn.NumOfCalls()));
   const MinimumState & initialState = result.back();
   
   
   double edm = initialState.Edm();
   
   
#ifdef DEBUG
   std::cout << "\n\nDEBUG Variable Metric Builder  \nInitial State: "  
             << " Parameter " << initialState.Vec()       
             << " Gradient " << initialState.Gradient().Vec() 
             << " Inv Hessian " << initialState.Error().InvHessian()  
             << " edm = " << initialState.Edm() << std::endl;
#endif
   
   
   
   // iterate until edm is small enough or max # of iterations reached
   edm *= (1. + 3.*initialState.Error().Dcovar());
   MnLineSearch lsearch;
   MnAlgebraicVector step(initialState.Gradient().Vec().size());
   // keep also prevStep
   MnAlgebraicVector prevStep(initialState.Gradient().Vec().size());
   
   do {   
      
      //     const MinimumState& s0 = result.back();
      MinimumState s0 = result.back();
      
      step = -1.*s0.Error().InvHessian()*s0.Gradient().Vec();
      
#ifdef DEBUG
      std::cout << "\n\n---> Iteration - " << result.size() 
                << "\nFval = " << s0.Fval() << " numOfCall = " << fcn.NumOfCalls() 
                << "\nInternal Parameter values " << s0.Vec() 
                << " Newton step " << step << std::endl; 
#endif
      
      
      double gdel = inner_product(step, s0.Gradient().Grad());
      if(gdel > 0.) {
#ifdef WARNINGMSG
         std::cout<<"VariableMetricBuilder: matrix not pos.def."<<std::endl;
         std::cout<<"gdel > 0: "<<gdel<<std::endl;
#endif
         MnPosDef psdf;
         s0 = psdf(s0, prec);
         step = -1.*s0.Error().InvHessian()*s0.Gradient().Vec();
         // #ifdef DEBUG
         //       std::cout << "After MnPosdef - Error  " << s0.Error().InvHessian() << " Gradient " << s0.Gradient().Vec() << " step " << step << std::endl;      
         // #endif
         gdel = inner_product(step, s0.Gradient().Grad());
#ifdef WARNINGMSG
         std::cout<<"gdel: "<<gdel<<std::endl;
#endif
         if(gdel > 0.) {
            result.push_back(s0);
            return FunctionMinimum(seed, result, fcn.Up());
         }
      }
      MnParabolaPoint pp = lsearch(fcn, s0.Parameters(), step, gdel, prec);
      if(fabs(pp.y() - s0.Fval()) < fabs(s0.Fval())*prec.Eps() ) {
#ifdef WARNINGMSG
         std::cout<<"VariableMetricBuilder: warning: no improvement in line search  " << std::endl;
#endif
         // no improvement exit   (is it really needed LM ? in vers. 1.22 tried alternative )
         // add new state where only fcn changes
         result.push_back(MinimumState(s0.Parameters(), s0.Error(), s0.Gradient(), s0.Edm(), fcn.NumOfCalls()) );
         break; 
         
         
      }
      
#ifdef DEBUG
      std::cout << "Result after line search : \nx = " << pp.x() 
                << "\nOld Fval = " << s0.Fval() 
                << "\nNew Fval = " << pp.y() 
                << "\nNFcalls = " << fcn.NumOfCalls() << std::endl; 
#endif
      
      MinimumParameters p(s0.Vec() + pp.x()*step, pp.y());
      
      
      FunctionGradient g = gc(p, s0.Gradient());
      
      
      edm = Estimator().Estimate(g, s0.Error());
      
      
      if(edm < 0.) {
         std::cout<<"VariableMetricBuilder: matrix not pos.def."<<std::endl;
         std::cout<<"edm < 0"<<std::endl;
         MnPosDef psdf;
         s0 = psdf(s0, prec);
         edm = Estimator().Estimate(g, s0.Error());
         if(edm < 0.) {
            result.push_back(s0);
            return FunctionMinimum(seed, result, fcn.Up());
         }
      } 
      MinimumError e = ErrorUpdator().Update(s0, p, g);
      
#ifdef DEBUG
      std::cout << "Updated new point: \n " 
                << " Parameter " << p.Vec()       
                << " Gradient " << g.Vec() 
                << " InvHessian " << e.Matrix() 
                << " Hessian " << e.Hessian() 
                << " edm = " << edm << std::endl << std::endl;
#endif
      
      
      result.push_back(MinimumState(p, e, g, edm, fcn.NumOfCalls())); 
      
      // correct edm 
      edm *= (1. + 3.*e.Dcovar());
      
#ifdef DEBUG
      std::cout << "edm corrected = " << edm << std::endl;
#endif
      
      
      
   } while(edm > edmval && fcn.NumOfCalls() < maxfcn);  // end of iteration loop
   
   if(fcn.NumOfCalls() >= maxfcn) {
#ifdef WARNINGMSG
      std::cout<<"VariableMetricBuilder: call limit exceeded."<<std::endl;
#endif
      return FunctionMinimum(seed, result, fcn.Up(), FunctionMinimum::MnReachedCallLimit());
   }
   
   if(edm > edmval) {
      if(edm < fabs(prec.Eps2()*result.back().Fval())) {
#ifdef WARNINGMSG
         std::cout<<"VariableMetricBuilder: machine accuracy limits further improvement."<<std::endl;
#endif
         return FunctionMinimum(seed, result, fcn.Up());
      } else if(edm < 10.*edmval) {
         return FunctionMinimum(seed, result, fcn.Up());
      } else {
#ifdef WARNINGMSG
         std::cout<<"VariableMetricBuilder: iterations finish without convergence."<<std::endl;
         std::cout<<"VariableMetricBuilder: edm= "<<edm<<" requested: "<<edmval<<std::endl;
#endif
         return FunctionMinimum(seed, result, fcn.Up(), FunctionMinimum::MnAboveMaxEdm());
      }
   }
   //   std::cout<<"result.back().Error().Dcovar()= "<<result.back().Error().Dcovar()<<std::endl;
   
#ifdef DEBUG
   std::cout << "Exiting succesfully Variable Metric Builder \n" 
             << "NFCalls = " << fcn.NumOfCalls() 
             << "\nFval = " <<  result.back().Fval() 
             << "\nedm = " << edm << " requested = " << edmval << std::endl; 
#endif
   
   return FunctionMinimum(seed, result, fcn.Up());
}

   }  // namespace Minuit2

}  // namespace ROOT
