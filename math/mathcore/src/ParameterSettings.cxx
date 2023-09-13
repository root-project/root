// @(#)root/mathcore:$Id$
// Author: L. Moneta Thu Sep 21 16:21:48 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class ParameterSettings

#include <Fit/ParameterSettings.h>

#include <Math/Error.h>

namespace ROOT {

namespace Fit {

/// set a double side limit,
/// if low == up the parameter is fixed  if low > up the limits are removed
/// The current parameter value should be within the given limits [low,up].
/// If the value is outside the limits, then a new parameter value is set to = (up+low)/2
void ParameterSettings::SetLimits(double low, double up)
{

   if (low > up) {
      RemoveLimits();
      return;
   }
   if (low == up && low == fValue) {
      Fix();
      return;
   }
   if (low > fValue || up < fValue) {
      MATH_INFO_MSG("ParameterSettings",
                    "lower/upper bounds outside current parameter value. The value will be set to (low+up)/2 ");
      fValue = 0.5 * (up + low);
   }
   fLowerLimit = low;
   fUpperLimit = up;
   fHasLowerLimit = true;
   fHasUpperLimit = true;
}

} // end namespace Fit

} // end namespace ROOT
