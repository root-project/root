// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnTiny.h"

namespace ROOT {

   namespace Minuit2 {


MnMachinePrecision::MnMachinePrecision() : 
   fEpsMac(4.0E-7),
   fEpsMa2(2.*sqrt(4.0E-7)) {
      
   //determine machine precision
   /*
       char e[] = {"e"};
       fEpsMac = 8.*dlamch_(e);
       fEpsMa2 = 2.*sqrt(fEpsMac);
   */
      
   //   std::cout<<"machine precision eps: "<<Eps()<<std::endl;
      
   MnTiny mytiny;
      
   //calculate machine precision
   double epstry = 0.5;
   double epsbak = 0.;
   double epsp1 = 0.;
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
