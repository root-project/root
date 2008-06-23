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
#include <cmath>

namespace ROOT { 

   namespace Math { 
      class Minimizer; 
   }


   namespace Fit { 

   class FitConfig; 

//___________________________________________________________________________________
/** 
   class containg the result of the fit and all the related information 
   (fitted parameter values, error, covariance matrix and minimizer result information)
   Contains a pointer also to the fitted (model) function, modified with the fit parameter values.  
   When the fit is valid, it is constructed from a  Minimizer and a model function pointer 

   @ingroup FitMain
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


   /** minimization quantities **/

   /// minimizer type 
   const std::string & MinimizerType() const { return fMinimType; } 

   /// True if fit successful, otherwise false.
   bool IsValid() const { return fValid; }
 
   /// Return value of the objective function (chi2 or likelihood) used in the fit
   double MinFcnValue() const { return fVal; } 

   ///Number of function calls to find minimum
   unsigned int NCalls() const { return fNCalls; }
   
   ///Expected distance from minimum 
   double Edm() const { return fEdm; }

   /** fitting quantities **/

   /// Return pointer to model (fit) function with fitted parameter values.
   const IModelFunction * FittedFunction() const { return fFitFunc; }

   /// Chi2 fit value
   /// in case of likelihood must be computed ? 
   double Chi2() const { return fChi2; } 

   /// Number of degree of freedom
   unsigned int Ndf() const { return fNdf; } 

   /// p value of the fit (chi2 probability)
   double Prob() const;  

 
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

   /// retrieve covariance matrix element 
   double CovMatrix (unsigned int i, unsigned int j) const { 
      if ( i >= fErrors.size() || j >= fErrors.size() ) return 0; 
      if (fCovMatrix.size() == 0) return 0; // no matrix is available in case of non-valid fits
      if ( j < i ) 
         return fCovMatrix[j + i* (i+1) / 2];
      else 
         return fCovMatrix[i + j* (j+1) / 2];
   }

   /// retrieve correlation elements 
   double Correlation(unsigned int i, unsigned int j ) const { 
      if ( i >= fErrors.size() || j >= fErrors.size() ) return 0; 
      if (fCovMatrix.size() == 0) return 0; // no matrix is available in case of non-valid fits
      double tmp = CovMatrix(i,i)*CovMatrix(j,j); 
      return ( tmp < 0) ? 0 : CovMatrix(i,j)/ std::sqrt(tmp); 
   }
   
   /// fill covariance matrix elements using a generic symmetric matrix class implementing operator(i,j)
   /// the matrix must be previously allocates with right size (npar * npar) 
   template<class Matrix> 
   void GetCovarianceMatrix(Matrix & mat) { 
      int npar = fErrors.size(); 
      for (int i = 0; i< npar; ++i) { 
         for (int j = 0; j<=i; ++i) { 
            mat(i,j) = fCovMatrix[j + i*(i+1)/2 ];
         }
      }
   }

   /// fill a correlation matrix elements using a generic symmetric matrix class implementing operator(i,j)
   /// the matrix must be previously allocates with right size (npar * npar) 
   template<class Matrix> 
   void GetCorrelationMatrix(Matrix & mat) { 
      int npar = fErrors.size(); 
      for (int i = 0; i< npar; ++i) { 
         for (int j = 0; j<=i; ++i) { 
            double tmp = fCovMatrix[i * (i +3)/2 ] * fCovMatrix[ j * (j+3)/2 ]; 
            if (tmp < 0) 
               mat(i,j) = 0; 
            else 
               mat(i,j) = fCovMatrix[j + i*(i+1)/2 ] / std::sqrt(tmp); 
         }
      }
   }


   /// get index for parameter name (return -1 if not found)
   int Index(const std::string & name) const; 


   ///normalize errors using chi2/ndf for chi2 fits
   void NormalizeErrors();

   /// flag to chek if errors are normalized
   bool NormalizedErrors() { return fNormalized; }


   /// print the result and optionaly covariance matrix and correlations
   void Print(std::ostream & os, bool covmat = false) const;

   ///print error matrix and correlations
   void PrintCovMatrix(std::ostream & os) const; 

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
