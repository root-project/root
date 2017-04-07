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
#include "Math/WrappedMultiTF1.h"

#include <iostream>

/** \class TFitResult
    \ingroup Hist
Extends the ROOT::Fit::Result class with a TNamed inheritance
providing easy possibility for I/O
*/

ClassImp(TFitResult);

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a ROOT::Fit::FitResult
/// copy the contained TF1 pointer function if it is

TFitResult::TFitResult(const ROOT::Fit::FitResult& f) : 
   TNamed("TFitResult","TFitResult"),
   ROOT::Fit::FitResult(f) 
{
   ROOT::Math::WrappedMultiTF1 * wfunc = dynamic_cast<ROOT::Math::WrappedMultiTF1 *>(ModelFunction().get() );
   if (wfunc)  wfunc->SetAndCopyFunction();
}


////////////////////////////////////////////////////////////////////////////////
/// Print result of the fit, by default chi2, parameter values and errors.
/// if option "V" is given print also error matrix and correlation

void TFitResult::Print(Option_t *option) const
{
   TString opt(option);
   opt.ToUpper();
   bool doCovMat = opt.Contains("V");
   ROOT::Fit::FitResult::Print( std::cout, doCovMat);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the covariance matrix from fit
///
/// The matrix is a symmetric matrix with a size N equal to
/// the total number of parameters considered in the fit including the fixed ones
/// The matrix row and columns corresponding to the fixed parameters will contain only zero's

TMatrixDSym TFitResult::GetCovarianceMatrix()  const
{
   if (CovMatrixStatus() == 0) {
      Warning("GetCovarianceMatrix","covariance matrix is not available");
      return TMatrixDSym();
   }
   TMatrixDSym mat(NPar());
   ROOT::Fit::FitResult::GetCovarianceMatrix<TMatrixDSym>(mat);
   return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the correlation matrix from fit.
///
/// The matrix is a symmetric matrix with a size N equal to
/// the total number of parameters considered in the fit including the fixed ones
/// The matrix row and columns corresponding to the fixed parameters will contain only zero's

TMatrixDSym TFitResult::GetCorrelationMatrix()  const
{
   if (CovMatrixStatus() == 0) {
      Warning("GetCorrelationMatrix","correlation matrix is not available");
      return TMatrixDSym();
   }
   TMatrixDSym mat(NPar());
   ROOT::Fit::FitResult::GetCorrelationMatrix<TMatrixDSym>(mat);
   return mat;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the TFitResult.

std::string cling::printValue(const TFitResult* val) {
   std::stringstream outs;
   val->ROOT::Fit::FitResult::Print(outs, false /*doCovMat*/);
   return outs.str();
}
