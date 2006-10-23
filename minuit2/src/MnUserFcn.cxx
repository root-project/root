// @(#)root/minuit2:$Name:  $:$Id: MnUserFcn.cxx,v 1.2 2006/07/03 22:06:42 moneta Exp $
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
