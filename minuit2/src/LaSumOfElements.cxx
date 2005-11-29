// @(#)root/minuit2:$Name:  $:$Id: LaSumOfElements.cpp,v 1.5.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/LAVector.h"
#include "Minuit2/LASymMatrix.h"

namespace ROOT {

   namespace Minuit2 {


double mndasum(unsigned int, const double*, int); 

double sum_of_elements(const LAVector& v) {
  
  return mndasum(v.size(), v.Data(), 1);
}

double sum_of_elements(const LASymMatrix& m) {
  
  return mndasum(m.size(), m.Data(), 1);
}

  }  // namespace Minuit2

}  // namespace ROOT
