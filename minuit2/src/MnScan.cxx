// @(#)root/minuit2:$Name:  $:$Id: MnScan.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnScan.h"
#include "Minuit2/MnParameterScan.h"

namespace ROOT {

   namespace Minuit2 {


std::vector<std::pair<double, double> > MnScan::Scan(unsigned int par, unsigned int maxsteps, double low, double high) {
   // perform a scan of the function in the parameter par
   MnParameterScan Scan(fFCN, fState.Parameters());
   double amin = Scan.Fval();
   
   std::vector<std::pair<double, double> > result = Scan(par, maxsteps, low, high);
   if(Scan.Fval() < amin) {
      fState.SetValue(par, Scan.Parameters().Value(par));
      amin = Scan.Fval();
   }
   
   return result;
}

   }  // namespace Minuit2

}  // namespace ROOT
