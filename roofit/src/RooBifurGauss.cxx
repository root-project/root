/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooBifurGauss.cc,v 1.2 2001/06/28 17:53:57 jback Exp $
 * Authors:
 *   Abi Soffer, Coloraro State University, abi@slac.stanford.edu
 * History:
 *   5-Dec-2000 Abi, Created.
 *  19-Jun-2001 JB, Ported to RooFitModels.
 *
 * Copyright (C) 2000 Coloraro State University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooBifurGauss.hh"
#include "RooFitCore/RooAbsReal.hh"

ClassImp(RooBifurGauss)

static const char rcsid[] =
"$Id: RooBifurGauss.cc,v 1.2 2001/06/28 17:53:57 jback Exp $";

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

RooBifurGauss::RooBifurGauss(const RooBifurGauss& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  sigmaL("sigmaL",this,other.sigmaL), sigmaR("sigmaR", this, other.sigmaR)
{
}

Double_t RooBifurGauss::evaluate(const RooArgSet* nset) const {

  Double_t arg = x - mean;

  Double_t coef(0.0);

  if (arg < 0.0){
    if (fabs(sigmaL) > 1e-30) {
      coef = -0.5/(sigmaL*sigmaL);
    }
  } else {
    if (fabs(sigmaR) > 1e-30) {
      coef = -0.5/(sigmaR*sigmaR);
    }
  }
    
  return exp(coef*arg*arg);
}

Int_t RooBifurGauss::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}



Double_t RooBifurGauss::analyticalIntegral(Int_t code) const 
{
  switch(code) {
  case 0: return getVal() ; 
  case 1: 
    {
      static Double_t root2 = sqrt(2) ;
      static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);
      
      Double_t coefL(0.0), coefR(0.0);
      if (fabs(sigmaL) > 1e-30) {
	coefL = -0.5/(sigmaL*sigmaL);
      }

      if (fabs(sigmaR) > 1e-30) {
	coefR = -0.5/(sigmaR*sigmaR);
      }

      Double_t xscaleL = root2*sigmaL;
      Double_t xscaleR = root2*sigmaR;

      return rootPiBy2*(sigmaR*erf((x.max() - mean)/xscaleR) - 
			sigmaL*erf((x.min() - mean)/xscaleL));
    }
  default: assert(0) ;
  }
}

