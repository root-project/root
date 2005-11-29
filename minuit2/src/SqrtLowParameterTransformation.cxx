// @(#)root/minuit2:$Name:  $:$Id: SqrtLowParameterTransformation.cpp,v 1.2.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

// Project   : LCG
// Package   : Minuit
// Author    : Lorenzo.MONETA@cern.ch 
// Created by: moneta  at Thu Apr  8 10:26:22 2004

#include "Minuit2/SqrtLowParameterTransformation.h"
#include "Minuit2/MnMachinePrecision.h"

namespace ROOT {

   namespace Minuit2 {


/// internal to external transformation
double SqrtLowParameterTransformation::Int2ext(double value, double lower) const {

  double val = lower - 1. + sqrt( value*value + 1.);
  return val; 
}

// external to internal transformation
double SqrtLowParameterTransformation::Ext2int(double value, double lower, const MnMachinePrecision& prec) const {
  
  double yy = value - lower + 1.; 
  double yy2 = yy*yy; 
  if (yy2 < (1. + prec.Eps2()) ) 
    return 8*sqrt(prec.Eps2()); 
  else 
    return sqrt( yy2 -1); 
}

// derivative of internal to external transofrmation
double SqrtLowParameterTransformation::DInt2Ext(double value, double) const {

  double val = value/( sqrt( value*value + 1.) );
  return val; 
}

  }  // namespace Minuit2

}  // namespace ROOT
