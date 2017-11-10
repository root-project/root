// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/

// Project   : LCG
// Package   : Minuit
// Author    : Lorenzo.MONETA@cern.ch
// Created by: moneta  at Thu Apr  8 10:26:22 2004


#include "Minuit2/SqrtUpParameterTransformation.h"
#include "Minuit2/MnMachinePrecision.h"

namespace ROOT {

  namespace Minuit2 {



    double SqrtUpParameterTransformation::Int2ext(double value, double upper) const {
      // internal to external transformation
      double val = upper + 1. - sqrt( value*value + 1.);
      return val;
    }


    double SqrtUpParameterTransformation::Ext2int(double value, double upper, const MnMachinePrecision& ) const {
      // external to internal transformation
      double yy = upper - value + 1.;
      double yy2 = yy*yy;
      if (yy2 < 1.  )
        return 0;
      else
        return sqrt( yy2 -1);
    }


    double SqrtUpParameterTransformation::DInt2Ext(double value, double) const {
      // derivative of internal to external transofrmation :  d (Int2Ext ) / d Int
      double val = - value/( sqrt( value*value + 1.) );
      return val;
    }


    double SqrtUpParameterTransformation::D2Int2Ext(double value, double) const {
      // second derivative of internal to external transformation :  d^2 (Int2Ext) / (d Int)^2
      double value_sq = value * value;
      return (value_sq / (value_sq + 1) - 1) / ( sqrt( value_sq + 1.) );
    }

    double SqrtUpParameterTransformation::GStepInt2Ext(double /*value*/, double /*upper*/) const {
      return 1.;
    }

  }  // namespace Minuit2

}  // namespace ROOT
