// @(#)root/minuit2:$Name:  $:$Id: MnGlobalCorrelationCoeff.cxx,v 1.3 2006/07/03 22:06:42 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnGlobalCorrelationCoeff.h"
#include <cmath>

#if defined(DEBUG) || defined(WARNINGMSG)
#include "Minuit2/MnPrint.h" 
#endif


namespace ROOT {

   namespace Minuit2 {


MnGlobalCorrelationCoeff::MnGlobalCorrelationCoeff(const MnAlgebraicSymMatrix& cov) : fGlobalCC(std::vector<double>()), fValid(true) {
   // constructor: calculate global correlation given a symmetric matrix 
   
   MnAlgebraicSymMatrix inv(cov);
   int ifail = Invert(inv);
   if(ifail != 0) {
#ifdef WARNINGMSG
      std::cout<<"MnGlobalCorrelationCoeff: inversion of matrix fails."<<std::endl;
#endif
      fValid = false;
   } else {
      
      for(unsigned int i = 0; i < cov.Nrow(); i++) {
         double denom = inv(i,i)*cov(i,i);
         if(denom < 1. && denom > 0.) fGlobalCC.push_back(0.);
         else fGlobalCC.push_back(std::sqrt(1. - 1./denom));
      }
   }
}

   }  // namespace Minuit2

}  // namespace ROOT
