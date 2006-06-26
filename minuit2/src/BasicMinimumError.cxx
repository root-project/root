// @(#)root/minuit2:$Name:  $:$Id: BasicMinimumError.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/BasicMinimumError.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {
   
   namespace Minuit2 {
      
      
      
MnAlgebraicSymMatrix BasicMinimumError::Hessian() const {
   // calculate Heassian: inverse of error matrix 
   MnAlgebraicSymMatrix tmp(fMatrix);
   int ifail = Invert(tmp);
   if(ifail != 0) {
      std::cout<<"BasicMinimumError inversion fails; return diagonal matrix."<<std::endl;
      MnAlgebraicSymMatrix tmp(fMatrix.Nrow());
      for(unsigned int i = 0; i < fMatrix.Nrow(); i++) {
         tmp(i,i) = 1./fMatrix(i,i);
      }
      return tmp;
   }
   return tmp;
}

   }  // namespace Minuit2
   
}  // namespace ROOT
