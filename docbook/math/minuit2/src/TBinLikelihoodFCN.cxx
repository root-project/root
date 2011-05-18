// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TBinLikelihoodFCN.h"
#include "TChi2FitData.h"
#include "TChi2FCN.h"
#include "FitterUtil.h"

#include <cmath>
#include <cassert>

#include "TF1.h"
#include "TVirtualFitter.h"

//#define DEBUG 1
#ifdef DEBUG
#include <iostream>
#endif



TBinLikelihoodFCN::TBinLikelihoodFCN( const TVirtualFitter & fitter) : 
  fUp(1.0), fOwner(true)
{ 
   // constructor (create fit data class) and keep a pointer to the model function
   // use errordef of 1 since we multiply the lokelihood by a factor of 2 
   fFunc = dynamic_cast<TF1 *> ( fitter.GetUserFunc() );
   assert(fFunc != 0);
   // to do: use class for likelihood data (errors are not necessary)
   // in likelihood fit need to keep empty bins
   fData = new TChi2FitData(fitter, false); 
#ifdef DEBUG
   std::cout << "Created FitData with size = " << fData->Size() << std::endl;
#endif
   
   // need to set the size so ROOT can calculate ndf.
   fFunc->SetNumberFitPoints(fData->Size());
}



TBinLikelihoodFCN::~TBinLikelihoodFCN() {  
//  if this class manages the fit data class, delete it at the end
   if (fOwner && fData) { 
#ifdef DEBUG
      std::cout << "deleting the data - size is " << fData->Size() << std::endl; 
#endif
      delete fData; 
   }
}



double TBinLikelihoodFCN::operator()(const std::vector<double>& par) const {
// implement log-likelihood function using the fit data and model function 
   
   assert(fData != 0); 
   assert(fFunc != 0); 
   
   // safety measure against negative logs
   static const double epsilon = 1e-300;
   
   //  std::cout << "number of params " << par.size() << " in TF1 " << fFunc->GetNpar() << "  " << fFunc->GetNumberFreeParameters() << std::endl;
   
   unsigned int n = fData->Size();
   double loglike = 0;
   int nRejected = 0; 
   for (unsigned int i = 0; i < n; ++ i) { 
      fFunc->RejectPoint(false); 
      const std::vector<double> & x = fData->Coords(i); 
      double y = fData->Value(i);
      //std::cout << x[0] << "  " << y << "  " << 1./invError << " params " << par[0] << std::endl;
      double fval;
      if (fData->UseIntegral()) { 
         const std::vector<double> & x2 = fData->Coords(i+1);
         fval = FitterUtil::EvalIntegral(fFunc,x,x2,par);
      }
      else   
         fval = fFunc->EvalPar( &x.front(), &par.front() ); 
      
      if (fFunc->RejectedPoint() ) { 
         nRejected++; 
         continue; 
      }
      
      double logtmp;
      // protections against negative argument to the log 
      // smooth linear extrapolation below pml_A
      if(fval<=epsilon) logtmp = fval/epsilon + std::log(epsilon) - 1; 
      else       logtmp = std::log(fval);
      
      loglike +=  fval - y*logtmp;  
   }
   
   // reset the number of fitting data points
   if (nRejected != 0)  fFunc->SetNumberFitPoints(n-nRejected);
   
   return 2.*loglike;
}


double TBinLikelihoodFCN::Chi2(const std::vector<double>& par) const {
// function to evaluate the chi2 equivalent 
   TChi2FCN chi2Fcn(fData,fFunc);
   return chi2Fcn(par);
}
