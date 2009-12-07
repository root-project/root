// @(#)root/hist:$Id$
// Author: David Gonzalez Maline   12/11/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TFitResult.h"

#include <iostream>

/**
TFitResult extends the ROOT::Fit::Result class with a TNamed inheritance
providing easy possibility for I/O

 */

ClassImp(TFitResult)

void TFitResult::Print(Option_t *option) const
{
   // print result of the fit, by default chi2, parameter values and errors 
   // if option "V" is given print also error matrix and correlation
   
   TString opt(option); 
   opt.ToUpper();
   bool doCovMat = opt.Contains("V");
   ROOT::Fit::FitResult::Print( std::cout, doCovMat); 
}

TMatrixDSym TFitResult::GetCovarianceMatrix()  const
{
   // Return the covariance matrix from fit
   // The matrix is a symmetric matrix with a size N equal to 
   // the total number of parameters considered in the fit including the fixed ones
   // The matrix row and columns corresponding to the fixed parameters will contain only zero's

   if (CovMatrixStatus() == 0) {
      Warning("GetCovarianceMatrix","covariance matrix is not available"); 
      return TMatrixDSym();
   }
   TMatrixDSym mat(NPar());
   ROOT::Fit::FitResult::GetCovarianceMatrix<TMatrixDSym>(mat);
   return mat; 
}

TMatrixDSym TFitResult::GetCorrelationMatrix()  const
{
   // Return the correlation matrix from fit. 
   // The matrix is a symmetric matrix with a size N equal to 
   // the total number of parameters considered in the fit including the fixed ones
   // The matrix row and columns corresponding to the fixed parameters will contain only zero's
   if (CovMatrixStatus() == 0) {
      Warning("GetCorrelationMatrix","correlation matrix is not available"); 
      return TMatrixDSym();
   }
   TMatrixDSym mat(NPar());
   ROOT::Fit::FitResult::GetCorrelationMatrix<TMatrixDSym>(mat);
   return mat; 
}


