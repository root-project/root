// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FunctionGradient
#define ROOT_Minuit2_FunctionGradient

#include "Minuit2/MnMatrix.h"

#include <memory>

namespace ROOT {

namespace Minuit2 {

class FunctionGradient {

private:
public:
   explicit FunctionGradient(unsigned int n)
      : fPtr{new Data{MnAlgebraicVector(n), MnAlgebraicVector(n), MnAlgebraicVector(n), false, false}}
   {
   }

   explicit FunctionGradient(const MnAlgebraicVector &grd)
      : fPtr{new Data{grd, MnAlgebraicVector(grd.size()), MnAlgebraicVector(grd.size()), true, true}}
   {
   }

   FunctionGradient(const MnAlgebraicVector &grd, const MnAlgebraicVector &g2, const MnAlgebraicVector &gstep)
      : fPtr{new Data{grd, g2, gstep, true, false}}
   {
   }

   const MnAlgebraicVector &Grad() const { return fPtr->fGradient; }
   const MnAlgebraicVector &Vec() const { return Grad(); }
   bool IsValid() const { return fPtr->fValid; }

   bool IsAnalytical() const { return fPtr->fAnalytical; }
   const MnAlgebraicVector &G2() const { return fPtr->fG2ndDerivative; }
   const MnAlgebraicVector &Gstep() const { return fPtr->fGStepSize; }

private:
   struct Data {
      MnAlgebraicVector fGradient;
      MnAlgebraicVector fG2ndDerivative;
      MnAlgebraicVector fGStepSize;
      bool fValid;
      bool fAnalytical;
   };

   std::shared_ptr<Data> fPtr;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_FunctionGradient
