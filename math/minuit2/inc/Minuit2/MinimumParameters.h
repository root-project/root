// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MinimumParameters
#define ROOT_Minuit2_MinimumParameters

#include "Minuit2/MnMatrix.h"

namespace ROOT {

namespace Minuit2 {

class MinimumParameters {

public:
   MinimumParameters(unsigned int n, double fval = 0)
      : fPtr{new Data{MnAlgebraicVector(n), MnAlgebraicVector(n), fval, false, false}}
   {
   }

   /** takes the Parameter vector */
   MinimumParameters(const MnAlgebraicVector &avec, double fval)
      : fPtr{new Data{avec, MnAlgebraicVector(avec.size()), fval, true, false}}
   {
   }

   /** takes the Parameter vector plus step size x1 - x0 = dirin */
   MinimumParameters(const MnAlgebraicVector &avec, const MnAlgebraicVector &dirin, double fval)
      : fPtr{new Data{avec, dirin, fval, true, true}}
   {
   }

   const MnAlgebraicVector &Vec() const { return fPtr->fParameters; }
   const MnAlgebraicVector &Dirin() const { return fPtr->fStepSize; }
   double Fval() const { return fPtr->fFVal; }
   bool IsValid() const { return fPtr->fValid; }
   bool HasStepSize() const { return fPtr->fHasStep; }

private:
   struct Data {
      MnAlgebraicVector fParameters;
      MnAlgebraicVector fStepSize;
      double fFVal;
      bool fValid;
      bool fHasStep;
   };

   std::shared_ptr<Data> fPtr;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MinimumParameters
