// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MinimumSeed
#define ROOT_Minuit2_MinimumSeed

#include "Minuit2/MinimumState.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MnUserTransformation.h"

namespace ROOT {

namespace Minuit2 {

class MinimumSeed {

public:
   MinimumSeed(const MinimumState &state, const MnUserTransformation &trafo) : fPtr{new Data{state, trafo, true}} {}

   const MinimumState &State() const { return fPtr->fState; }
   const MinimumParameters &Parameters() const { return State().Parameters(); }
   const MinimumError &Error() const { return State().Error(); };
   const FunctionGradient &Gradient() const { return State().Gradient(); }
   const MnUserTransformation &Trafo() const { return fPtr->fTrafo; }
   const MnMachinePrecision &Precision() const { return Trafo().Precision(); }
   double Fval() const { return State().Fval(); }
   double Edm() const { return State().Edm(); }
   unsigned int NFcn() const { return State().NFcn(); }
   bool IsValid() const { return fPtr->fValid; }

private:
   struct Data {
      MinimumState fState;
      MnUserTransformation fTrafo;
      bool fValid;
   };

   std::shared_ptr<Data> fPtr;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MinimumSeed
