// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:05:34 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class FitResult

#include "Fit/FitResult.h"

#include "Fit/FitConfig.h"

#include "Math/Minimizer.h"

#include "Math/IParamFunction.h"

#include "Math/DistFunc.h"

#include <cassert>
#include <cmath>

namespace ROOT { 

   namespace Fit { 


FitResult::FitResult() : 
   fValid(false), fNormalized(false), fVal(0), fEdm(0), fChi2(0), fNdf(0), fNCalls(0), fDataSize(0), fFitFunc(0)
{
   // Default constructor implementation.
}

      FitResult::FitResult(ROOT::Math::Minimizer & min, const FitConfig & fconfig, const IModelFunction & func,  bool isValid,  unsigned int sizeOfData, const  ROOT::Math::IMultiGenFunction * chi2func, bool minosErr, unsigned int ncalls ) : 
   fValid(isValid),
   fNormalized(false),
   fVal (min.MinValue()),  
   fEdm (min.Edm()),  
   fNCalls(min.NCalls()),
   fParams(std::vector<double>(min.X(), min.X() + min.NDim() ) ),  
   fFitFunc(&func)
{
   // Constructor from a minimizer, fill the data

   if (sizeOfData > 0) fNdf = sizeOfData - min.NFree(); 

   if (min.Errors() != 0) 
      fErrors = std::vector<double>(min.Errors(), min.Errors() + min.NDim() ) ; 

   if (chi2func == 0) 
      fChi2 = fVal;
   else { 
      // compute chi2 equivalent
      fChi2 = (*chi2func)(&fParams[0]); 
   }

   // replace ncalls if given (they are taken from the FitMethodFunction)
   if (ncalls !=0) fNCalls = ncalls;

   unsigned int n  = min.NDim(); 
      
//    // fill error matrix
   // cov matrix rank 
   if (fValid) { 
      unsigned int r = n * (  n + 1 )/2;  
      fCovMatrix.reserve(r);
      for (unsigned int i = 0; i < n; ++i) 
         for (unsigned int j = 0; j <= i; ++j)
            fCovMatrix.push_back(min.CovMatrix(i,j) );
      
      assert (fCovMatrix.size() == r ); 
   }

   // minos errors 
   if (minosErr) { 
      fMinosErrors.reserve(n);
      for (unsigned int i = 0; i < n; ++i) { 
         double elow, eup; 
         bool ret = min.GetMinosError(0, elow, eup); 
         if (ret) fMinosErrors.push_back(std::make_pair(elow,eup) );
         else fMinosErrors.push_back(std::make_pair(0.,0.) );
      }
   }

   fMinimType = fconfig.MinimizerType();
   if (fconfig.MinimizerAlgoType() != "") fMinimType += " / " + fconfig.MinimizerAlgoType(); 
}

void FitResult::NormalizeErrors() { 
   // normalize errors and covariance matrix according to chi2 value
   if (fNdf == 0 || fChi2 <= 0) return; 
   double s2 = fChi2/fNdf; 
   double s = std::sqrt(fChi2/fNdf); 
   for (unsigned int i = 0; i < fErrors.size() ; ++i) 
      fErrors[i] *= s; 
   for (unsigned int i = 0; i < fCovMatrix.size() ; ++i) 
      fCovMatrix[i] *= s2; 

   fNormalized = true; 
} 

double FitResult::Prob() const { 
   // fit probability
   return ROOT::Math::chisquared_cdf_c(fChi2, static_cast<double>(fNdf) ); 
}

int FitResult::Index(const std::string & name) const { 
   // find index for given parameter name
   unsigned int npar = fParams.size(); 
   for (unsigned int i = 0; i < npar; ++i) 
      if ( fFitFunc->ParameterName(i) == name) return i; 
   
   return -1; // case name is not found
} 

void FitResult::Print(std::ostream & os) const { 
   // print the result in the given stream 
   if (!fValid) { 
      os << "\n****************************************\n";
      os << "            Invalid FitResult            ";
      os << "\n****************************************\n";
      return; 
   }

   os << "\n****************************************\n";
   os << "            FitResult                   \n\n";
   os << "Minimizer is " << fMinimType << std::endl;
   unsigned int npar = fParams.size(); 
   os << "Chi2/Likelihood  =\t" << fVal << std::endl;
   if (fVal != fChi2) 
   os << "Chi2             =\t" << fChi2<< std::endl;
   os << "NDf              =\t" << fNdf << std::endl; 
   os << "Edm              =\t" << fEdm << std::endl; 
   os << "NCalls           =\t" << fNCalls << std::endl; 
   assert(fFitFunc != 0); 
   for (unsigned int i = 0; i < npar; ++i) { 
      os << fFitFunc->ParameterName(i) << "\t\t =\t" << fParams[i] << " \t+/-\t" << fErrors[i] << std::endl; 
   }
}

   } // end namespace Fit

} // end namespace ROOT

