// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MinimumError
#define ROOT_Minuit2_MinimumError

#include "Minuit2/MnConfig.h"
#include "Minuit2/MnMatrix.h"
#include "Minuit2/MnPrint.h"

#include <memory>

namespace ROOT {

namespace Minuit2 {

/** MinimumError keeps the inv. 2nd derivative (inv. Hessian) used for
    calculating the Parameter step size (-V*g) and for the covariance Update
    (ErrorUpdator). The covariance matrix is equal to twice the inv. Hessian.
 */
class MinimumError {

public:
   enum Status {
      MnUnset,
      MnPosDef,
      MnMadePosDef,
      MnNotPosDef,
      MnHesseFailed,
      MnInvertFailed,
      MnReachedCallLimit,
   };

public:
   MinimumError(unsigned int n) : fPtr{new Data{{n}, 1.0, MnUnset}} {}

   MinimumError(const MnAlgebraicSymMatrix &mat, double dcov) : fPtr{new Data{mat, dcov, MnPosDef}} {}

   MinimumError(const MnAlgebraicSymMatrix &mat, Status status) : fPtr{new Data{mat, 1.0, status}} {}

   MnAlgebraicSymMatrix Matrix() const { return 2. * fPtr->fMatrix; } // why *2 ?

   const MnAlgebraicSymMatrix &InvHessian() const { return fPtr->fMatrix; }

   // calculate invert of matrix. Used to compute Hessian  by inverting matrix
   MnAlgebraicSymMatrix Hessian() const
   {
      return InvertMatrix(fPtr->fMatrix);
   }

   static MnAlgebraicSymMatrix InvertMatrix(const MnAlgebraicSymMatrix & matrix, int & ifail) {
       // calculate inverse of given matrix
      MnAlgebraicSymMatrix tmp(matrix);
      ifail = ROOT::Minuit2::Invert(tmp);
      if (ifail != 0) {
         MnPrint print("MinimumError::Invert");
         print.Warn("Inversion fails; return diagonal matrix");
         for (unsigned int i = 0; i < matrix.Nrow(); ++i)
            for (unsigned int j = 0; j <= i; j++)
               tmp(i, j) = i == j ? 1. / matrix(i, i) : 0;
      }
      return tmp;
   }
   static MnAlgebraicSymMatrix InvertMatrix(const MnAlgebraicSymMatrix & matrix) {
      int ifail = 0;
      return InvertMatrix(matrix, ifail);
   }

   double Dcovar() const { return fPtr->fDCovar; }
   Status GetStatus() const { return fPtr->fStatus; }

   bool IsValid() const { return IsAvailable() && (IsPosDef() || IsMadePosDef() || IsNotPosDef()); }
   bool IsAccurate() const { return IsPosDef() && Dcovar() < 0.1; }

   bool IsPosDef() const { return GetStatus() == MnPosDef; }
   bool IsMadePosDef() const { return GetStatus() == MnMadePosDef; }
   bool IsNotPosDef() const { return GetStatus() == MnNotPosDef; }
   bool HesseFailed() const { return GetStatus() == MnHesseFailed; }
   bool InvertFailed() const { return GetStatus() == MnInvertFailed; }
   bool HasReachedCallLimit() const { return GetStatus() == MnReachedCallLimit; }
   bool IsAvailable() const { return GetStatus() != MnUnset; }

private:
   struct Data {
      MnAlgebraicSymMatrix fMatrix;
      double fDCovar;
      Status fStatus;
   };

   std::shared_ptr<Data> fPtr;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MinimumError
