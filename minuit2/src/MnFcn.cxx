// @(#)root/minuit2:$Name:  $:$Id: MnFcn.cpp,v 1.1.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnFcn.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnVectorTransform.h"

namespace ROOT {

   namespace Minuit2 {


MnFcn::~MnFcn() {
//   std::cout<<"Total number of calls to FCN: "<<fNumCall<<std::endl;
}

double MnFcn::operator()(const MnAlgebraicVector& v) const {

  fNumCall++;
  return fFCN(MnVectorTransform()(v));
}

// double MnFcn::operator()(const std::vector<double>& par) const {
//     return fFCN(par);
// }

double MnFcn::ErrorDef() const {return fFCN.Up();}

double MnFcn::Up() const {return fFCN.Up();}

  }  // namespace Minuit2

}  // namespace ROOT
