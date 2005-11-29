// @(#)root/minuit2:$Name:  $:$Id: SinParameterTransformation.cpp,v 1.7.6.4 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/SinParameterTransformation.h"
#include "Minuit2/MnMachinePrecision.h"

#include <math.h>

namespace ROOT {

   namespace Minuit2 {


double SinParameterTransformation::Int2ext(double Value, double Upper, double Lower) const {
  
  return Lower + 0.5*(Upper - Lower)*(sin(Value) + 1.);
}

double SinParameterTransformation::Ext2int(double Value, double Upper, double Lower, const MnMachinePrecision& prec) const {

  double piby2 = 2.*atan(1.);
  double distnn = 8.*sqrt(prec.Eps2());
  double vlimhi = piby2 - distnn;
  double vlimlo = -piby2 + distnn;
  
  double yy = 2.*(Value - Lower)/(Upper - Lower) - 1.;
  double yy2 = yy*yy;
  if(yy2 > (1. - prec.Eps2())) {
    if(yy < 0.) {
      // Lower limit
//       std::cout<<"SinParameterTransformation warning: is at its Lower allowed limit. "<<Value<<std::endl;
      return vlimlo;
    } else {
      // Upper limit
//       std::cout<<"SinParameterTransformation warning: is at its Upper allowed limit."<<std::endl;
      return vlimhi;
    }
    
  } else {
    return asin(yy); 
  }
}

double SinParameterTransformation::DInt2Ext(double Value, double Upper, double Lower) const {

  return 0.5*fabs((Upper - Lower)*cos(Value));
}

  }  // namespace Minuit2

}  // namespace ROOT
