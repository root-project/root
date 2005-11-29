// @(#)root/minuit2:$Name:  $:$Id: MnUserFcn.h,v 1.2.6.2 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnUserFcn
#define ROOT_Minuit2_MnUserFcn

#include "Minuit2/MnFcn.h"

namespace ROOT {

   namespace Minuit2 {


class MnUserTransformation;

class MnUserFcn : public MnFcn {

public:

  MnUserFcn(const FCNBase& fcn, const MnUserTransformation& trafo) :
    MnFcn(fcn), fTransform(trafo) {}

  ~MnUserFcn() {}

  virtual double operator()(const MnAlgebraicVector&) const;

private:

  const MnUserTransformation& fTransform;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnUserFcn
