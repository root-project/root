// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnMachinePrecision.h"
#include <limits>

namespace ROOT {

namespace Minuit2 {

MnMachinePrecision::MnMachinePrecision()
{
   // use double precision values from the numeric_limits standard
   // and do not determine it anymore using ComputePrecision
   // epsilon from stundard
   // note that there is a factor of 2 in the definition of
   // std::numeric_limitys::epsilon w.r.t DLAMCH epsilon

   fEpsMac = 4. * std::numeric_limits<double>::epsilon();
   fEpsMa2 = 2. * std::sqrt(fEpsMac);
}
void MnMachinePrecision::ComputePrecision()
{
   fEpsMac = 4.0E-7;
   fEpsMa2 = 2. * std::sqrt(fEpsMac);

   // determine machine precision using
   // code similar to DLAMCH LAPACK Fortran function
   /*
       char e[] = {"e"};
       fEpsMac = 8.*dlamch_(e);
       fEpsMa2 = 2.*std::sqrt(fEpsMac);
   */

   // calculate machine precision
   double epstry = 0.5;
   double epsbak = 0.;
   volatile double epsp1 = 0.; // allow to run this method with fast-math
   double one = 1.0;
   for (int i = 0; i < 100; i++) {
      epstry *= 0.5;
      epsp1 = one + epstry;
      epsbak = Tiny(epsp1);
      if (epsbak < epstry) {
         fEpsMac = 8. * epstry;
         fEpsMa2 = 2. * std::sqrt(fEpsMac);
         break;
      }
   }
}

double MnMachinePrecision::One() const
{
   return fOne;
}

double MnMachinePrecision::Tiny(double epsp1) const
{
   // evaluate minimal difference between two floating points
   double result = epsp1 - One();
   return result;
}

} // namespace Minuit2

} // namespace ROOT
