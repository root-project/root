/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooGaussian.cc,v 1.11 2001/10/03 16:17:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jan-2000 DK Created initial version from RooGaussianProb
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

#include "BaBar/BaBar.hh"
#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooGaussian.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooGaussian)

RooGaussian::RooGaussian(const char *name, const char *title,
			 RooAbsReal& _x, RooAbsReal& _mean,
			 RooAbsReal& _sigma) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
}


RooGaussian::RooGaussian(const RooGaussian& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}


Double_t RooGaussian::evaluate() const
{
  Double_t arg= x - mean;  
  return exp(-0.5*arg*arg/(sigma*sigma)) ;
}



Int_t RooGaussian::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}



Double_t RooGaussian::analyticalIntegral(Int_t code) const 
{
  assert(code==1) ;

  static Double_t root2 = sqrt(2) ;
  static Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);
  
  Double_t xscale = root2*sigma;
  return rootPiBy2*sigma*(erf((x.max()-mean)/xscale)-erf((x.min()-mean)/xscale));
}




Int_t RooGaussian::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars) const
{
  //if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}


void RooGaussian::generateEvent(Int_t code)
{
  assert(code==1) ;
  Double_t xgen ;
  while(1) {    
    xgen = RooRandom::randomGenerator()->Gaus(mean,sigma);
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      return ;
    }
  }
}


