// @(#)root/minuit2:$Name:  $:$Id: LaVtMVSimilarity.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/LASymMatrix.h"
#include "Minuit2/LAVector.h"
#include "Minuit2/LaProd.h"

namespace ROOT {

   namespace Minuit2 {


double mnddot(unsigned int, const double*, int, const double*, int);

double similarity(const LAVector& avec, const LASymMatrix& mat) {
   // calculate the similarity vector-matrix product: V^T M V
   // use matrix product and then dot function (using mnddot)
   
   LAVector tmp = mat*avec;
   
   double Value = mnddot(avec.size(), avec.Data(), 1, tmp.Data(), 1);
   return Value;
}

   }  // namespace Minuit2

}  // namespace ROOT
