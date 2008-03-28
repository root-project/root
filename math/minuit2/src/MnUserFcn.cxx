// @(#)root/minuit2:$Id$
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
   // call Fcn function transforming from a MnAlgebraicVector of internal values to a std::vector of external ones 
   fNumCall++;
   return Fcn()( fTransform(v) );
}

   }  // namespace Minuit2

}  // namespace ROOT
