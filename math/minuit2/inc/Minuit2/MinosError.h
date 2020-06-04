// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MinosError
#define ROOT_Minuit2_MinosError

#include "Minuit2/MnCross.h"
#include <iostream>
#include <utility>

namespace ROOT {

   namespace Minuit2 {

//____________________________________________________________________________________
/**
   Class holding the result of Minos (lower and upper values) for a specific parameter
 */

class MinosError {

public:

   MinosError() : fParameter(0), fMinValue(0.), fUpper(MnCross()), fLower(MnCross()) {}

   MinosError(unsigned int par, double min, const MnCross& low, const MnCross& up) : fParameter(par), fMinValue(min), fUpper(up), fLower(low) {}

   ~MinosError() {}

   MinosError(const MinosError& err) : fParameter(err.fParameter), fMinValue(err.fMinValue), fUpper(err.fUpper),  fLower(err.fLower) {}

   MinosError& operator=(const MinosError& ) = default;

   MinosError& operator()(const MinosError& err) {
      fParameter = err.fParameter;
      fMinValue = err.fMinValue;
      fUpper = err.fUpper;
      fLower = err.fLower;
      return *this;
   }

   std::pair<double,double> operator()() const {
      return std::pair<double,double>(Lower(), Upper());
   }
   double Lower() const {
      if ( AtLowerLimit() ) return LowerState().Parameter( Parameter() ).LowerLimit() -  fMinValue;
      return -1.*LowerState().Error(Parameter())*(1. + fLower.Value());
   }
   double Upper() const {
      if ( AtUpperLimit() ) return UpperState().Parameter( Parameter() ).UpperLimit() -  fMinValue;
      return UpperState().Error(Parameter())*(1. + fUpper.Value());
   }
   unsigned int Parameter() const {return fParameter;}
   const MnUserParameterState& LowerState() const {return fLower.State();}
   const MnUserParameterState& UpperState() const {return fUpper.State();}
   bool IsValid() const {return fLower.IsValid() && fUpper.IsValid();}
   bool LowerValid() const {return fLower.IsValid();}
   bool UpperValid() const {return fUpper.IsValid();}
   bool AtLowerLimit() const {return fLower.AtLimit();}
   bool AtUpperLimit() const {return fUpper.AtLimit();}
   bool AtLowerMaxFcn() const {return fLower.AtMaxFcn();}
   bool AtUpperMaxFcn() const {return fUpper.AtMaxFcn();}
   bool LowerNewMin() const {return fLower.NewMinimum();}
   bool UpperNewMin() const {return fUpper.NewMinimum();}
   unsigned int NFcn() const {return fUpper.NFcn() + fLower.NFcn();}
   double Min() const {return fMinValue;}

private:

   unsigned int fParameter;
   double fMinValue;
   MnCross fUpper;
   MnCross fLower;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MinosError
