// @(#)root/minuit2:$Name:  $:$Id: MnUserFcn.cpp,v 1.2.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnUserFcn.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnUserTransformation.h"

namespace ROOT {

   namespace Minuit2 {


double MnUserFcn::operator()(const MnAlgebraicVector& v) const {

  fNumCall++;
  return Fcn()( fTransform(v) );
}

  }  // namespace Minuit2

}  // namespace ROOT
