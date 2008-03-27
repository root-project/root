// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:05:34 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitResult

#ifndef ROOT_Fit_FitResult
#define ROOT_Fit_FitResult

#ifndef ROOT_Fit_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif
#ifndef ROOT_Fit_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif

#include <vector>
#include <string>

namespace ROOT { 

   namespace Math { 
      class Minimizer; 
   }


   namespace Fit { 

   class FitConfig; 

//___________________________________________________________________________________
/** 
   FitResult class containg the result of the fit.  
   Contains a reference to the fitted function. 
   When the fit is valid it is constraucted from a  Minimizer and a Model function pointer 
*/ 
class FitResult {

public: 

   typedef  ROOT::Math::IParamMultiFunction IModelFunction; 

   /** 
      Default constructor for an empty (non valid) fit result
   */ 
   FitResult (); 

   /**
      Construct from a Minimizer instance 
    */
   FitResult(ROOT::Math::Minimizer & min, const FitConfig & fconfig, const IModelFunction & f, bool isValid, unsigned int sizeOfData = 0, const ROOT::Math::IMultiGenFunction * chi2func = 0, bool minosErr = false, unsigned int ncalls = 0);

  // use default copy constructor and assignment operator

   /** 
      Destructor (no operations)
   */ 
   ~FitResult ()  {}  


public: 

   ///normalize errors using chi2/ndf for chi2 fits
   void NormalizeErrors();

   /// flag to chek if errors are normalized
   bool NormalizedErrors() { return fNormalized; }

   /// True if fit successful, otherwise false.
   bool IsValid() const { return fValid; }


   /// Return pointer to model (fit) function with fitted parameter values.
   const IModelFunction * FittedFunction() const { return fFitFunc; }

   /// Return value of the objective function (chi2 or likelihood) used in the fit
   double MinFcnValue() const { return fVal; } 

   /// Chi2 fit value
   /// in case of likelihood must be computed ? 
   double Chi2() const { return fChi2; } 

   /// Number of degree of freedom
   unsigned int Ndf() const { return fNdf; } 

   /// p value of the fit (chi2 probability)
   double Prob() const;  

   /// retrieve covariance matrix element 
   double CovMatrix (unsigned int i, unsigned int j) const { 
      if ( i >= fErrors.size() || j >= fErrors.size() ) return 0; 
      if (fCovMatrix.size() == 0) return 0; // nomatrix available in case of non-valid fits
      if ( j < i ) 
         return fCovMatrix[j + i* (i+1) / 2];
      else 
         return fCovMatrix[i + j* (j+1) / 2];
   }
 
   /// parameter errors
   const std::vector<double> & Errors() const { return fErrors; }

   /// parameter values
   const std::vector<double> & Parameters() const { return fParams; }

   /// parameter value by index
   double Value(unsigned int i) const { return fParams[i]; }

   /// parameter error by index
   double Error(unsigned int i) const { return fErrors[i]; } 

//    /// Minos  Errors 
//    const std::vector<std::pair<double, double> > MinosErrors() const; 

   /// lower Minos error
   double LowerError(unsigned int i) const { return fMinosErrors[i].first; } 

   /// upper Minos error
   double UpperError(unsigned int i) const { return fMinosErrors[i].second; }  

   /// get index for parameter name (return -1 if not found)
   int Index(const std::string & name) const; 

   /// print the result 
   void Print(std::ostream & os) const;

protected: 


private: 

   bool fValid; 
   bool fNormalized;
   double fVal; 
   double fEdm; 
   double fChi2;
   std::vector<double> fCov; 
   unsigned int fNdf; 
   unsigned int fNCalls; 
   std::vector<double> fParams; 
   std::vector<double> fErrors; 
   std::vector<double> fCovMatrix; 
   std::vector<std::pair<double,double> > fMinosErrors; 

   unsigned int fDataSize; 
   const IModelFunction * fFitFunc; 
   std::string fMinimType; 

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_FitResult */
