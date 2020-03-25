// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnTiny.h"
#include <limits>

namespace ROOT {

   namespace Minuit2 {


MnMachinePrecision::MnMachinePrecision() {
   // use double precision values from the numeric_limits standard
   // and do not determine it anymore using ComputePrecision
   fEpsMac = 8 * std::numeric_limits<double>::epsilon();
   fEpsMa2 = 2.*sqrt(fEpsMac);
}
void MnMachinePrecision::ComputePrecision() {
   fEpsMac = 4.0E-7;
   fEpsMa2 = 2.*sqrt(fEpsMac);

   //determine machine precision
   /*
       // use DLAMACH LAPACK function
       char e[] = {"e"};
       fEpsMac = 8.*dlamch_(e);
       fEpsMa2 = 2.*sqrt(fEpsMac);
   */

   MnTiny mytiny;

   //calculate machine precision
   double epstry = 0.5;
   double epsbak = 0.;
   volatile double epsp1 = 0.; // allow to run this method with fast-math
   double one = 1.0;
   for(int i = 0; i < 100; i++) {
      epstry *= 0.5;
      epsp1 = one + epstry;
      epsbak = mytiny(epsp1);
      if(epsbak < epstry) {
         fEpsMac = 8.*epstry;
         fEpsMa2 = 2.*sqrt(fEpsMac);
         break;
      }
   }

}

   }  // namespace Minuit2

}  // namespace ROOT
