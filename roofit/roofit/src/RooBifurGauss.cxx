/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   Abi Soffer, Colorado State University, abi@slac.stanford.edu            *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          Colorado State University                        *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#include "RooFit.h"

/** \class RooBifurGauss
    \ingroup Roofit

Bifurcated Gaussian p.d.f with different widths on left and right
side of maximum value
**/

#include "RooBifurGauss.h"

#include "RooAbsReal.h"
#include "RooMath.h"
#include "BatchHelpers.h"
#include "RooVDTHeaders.h"

#include "TMath.h"

#include <cmath>

using namespace std;

ClassImp(RooBifurGauss);

////////////////////////////////////////////////////////////////////////////////

RooBifurGauss::RooBifurGauss(const char *name, const char *title,
              RooAbsReal& _x, RooAbsReal& _mean,
              RooAbsReal& _sigmaL, RooAbsReal& _sigmaR) :
  RooAbsPdf(name, title),
  x     ("x"     , "Dependent"  , this, _x),
  mean  ("mean"  , "Mean"       , this, _mean),
  sigmaL("sigmaL", "Left Sigma" , this, _sigmaL),
  sigmaR("sigmaR", "Right Sigma", this, _sigmaR)

{
}

////////////////////////////////////////////////////////////////////////////////

RooBifurGauss::RooBifurGauss(const RooBifurGauss& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  sigmaL("sigmaL",this,other.sigmaL), sigmaR("sigmaR", this, other.sigmaR)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBifurGauss::evaluate() const {
  Double_t arg = x - mean;

  Double_t coef(0.0);

  if (arg < 0.0){
    if (TMath::Abs(sigmaL) > 1e-30) {
      coef = -0.5/(sigmaL*sigmaL);
    }
  } else {
    if (TMath::Abs(sigmaR) > 1e-30) {
      coef = -0.5/(sigmaR*sigmaR);
    }
  }

  return exp(coef*arg*arg);
}

////////////////////////////////////////////////////////////////////////////////

namespace BifurGaussBatchEvaluate {
//Author: Emmanouil Michalainas, CERN 20 AUGUST 2019  

template<class Tx, class Tm, class Tsl, class Tsr>
void compute(  size_t batchSize,
               double * __restrict__ output,
               Tx X, Tm M, Tsl SL, Tsr SR)
{
  for (size_t i=0; i<batchSize; i++) {
    const double arg = X[i]-M[i];
    output[i] = arg / ((arg < 0.0)*SL[i] + (arg >= 0.0)*SR[i]);
  }
  
  for (size_t i=0; i<batchSize; i++) {
    if (X[i]-M[i]>1e-30 || X[i]-M[i]<-1e-30) {
      output[i] = vdt::fast_exp(-0.5*output[i]*output[i]);
    }
    else {
      output[i] = 1.0;
    }
  }
}
};

RooSpan<double> RooBifurGauss::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  using namespace BifurGaussBatchEvaluate;
    
  EvaluateInfo info = getInfo( {&x,&mean,&sigmaL,&sigmaR}, begin, batchSize );
  auto output = _batchData.makeWritableBatchUnInit(begin, info.size);
  auto xData = x.getValBatch(begin, info.size);
  if (info.nBatches == 0) {
    throw std::logic_error("Requested a batch computation, but no batch data available.");
  }
  else if (info.nBatches==1 && !xData.empty()) {
    compute(info.size, output.data(), xData.data(), 
    BracketAdapter<double> (mean), 
    BracketAdapter<double> (sigmaL), 
    BracketAdapter<double> (sigmaR));
  }
  else {
    compute(info.size, output.data(), 
    BracketAdapterWithMask (x,x.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (mean,mean.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (sigmaL,sigmaL.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (sigmaR,sigmaR.getValBatch(begin,batchSize)));
  }
  return output;
}


////////////////////////////////////////////////////////////////////////////////

Int_t RooBifurGauss::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBifurGauss::analyticalIntegral(Int_t code, const char* rangeName) const
{
  switch(code) {
  case 1:
    {
      static Double_t root2 = sqrt(2.) ;
      static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);

//       Double_t coefL(0.0), coefR(0.0);
//       if (TMath::Abs(sigmaL) > 1e-30) {
//    coefL = -0.5/(sigmaL*sigmaL);
//       }

//       if (TMath::Abs(sigmaR) > 1e-30) {
//    coefR = -0.5/(sigmaR*sigmaR);
//       }

      Double_t xscaleL = root2*sigmaL;
      Double_t xscaleR = root2*sigmaR;

      Double_t integral = 0.0;
      if(x.max(rangeName) < mean)
      {
   integral = sigmaL * ( RooMath::erf((x.max(rangeName) - mean)/xscaleL) - RooMath::erf((x.min(rangeName) - mean)/xscaleL) );
      }
      else if (x.min(rangeName) > mean)
      {
   integral = sigmaR * ( RooMath::erf((x.max(rangeName) - mean)/xscaleR) - RooMath::erf((x.min(rangeName) - mean)/xscaleR) );
      }
      else
      {
   integral = sigmaR*RooMath::erf((x.max(rangeName) - mean)/xscaleR) - sigmaL*RooMath::erf((x.min(rangeName) - mean)/xscaleL);
      }
      //      return rootPiBy2*(sigmaR*RooMath::erf((x.max(rangeName) - mean)/xscaleR) -
      //         sigmaL*RooMath::erf((x.min(rangeName) - mean)/xscaleL));
      return integral*rootPiBy2;
    }
  }

  assert(0) ;
  return 0 ; // to prevent compiler warnings
}
