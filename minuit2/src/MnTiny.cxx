// @(#)root/minuit2:$Name:  $:$Id: MnTiny.cpp,v 1.1.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnTiny.h"

namespace ROOT {

   namespace Minuit2 {


double MnTiny::One() const {return fOne;}

double MnTiny::operator()(double epsp1) const {
  double result = epsp1 - One();
  return result;
}

  }  // namespace Minuit2

}  // namespace ROOT
