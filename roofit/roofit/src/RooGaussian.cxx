/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Plain Gaussian p.d.f
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooGaussian.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"

ClassImp(RooGaussian)


//_____________________________________________________________________________
RooGaussian::RooGaussian(const char *name, const char *title,
			 RooAbsReal& _x, RooAbsReal& _mean,
			 RooAbsReal& _sigma) :
  RooAbsPdf(name,title),
  x("x","Observable",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
}



//_____________________________________________________________________________
RooGaussian::RooGaussian(const RooGaussian& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}



//_____________________________________________________________________________
Double_t RooGaussian::evaluate() const
{
  Double_t arg= x - mean;  
  Double_t sig = sigma ;
  Double_t ret =exp(-0.5*arg*arg/(sig*sig)) ;
//   cout << "gauss(" << GetName() << ") x = " << x << " mean = " << mean << " sigma = " << sigma << " ret = " << ret << endl ;
  return ret ;
}



//_____________________________________________________________________________
Int_t RooGaussian::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  if (matchArgs(allVars,analVars,mean)) return 2 ;
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooGaussian::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  assert(code==1 || code==2) ;

  static const Double_t root2 = sqrt(2.) ;
  static const Double_t rootPiBy2 = sqrt(atan2(0.0,-1.0)/2.0);
  Double_t xscale = root2*sigma;
  Double_t ret = 0;
  if(code==1){  
    ret = rootPiBy2*sigma*(RooMath::erf((x.max(rangeName)-mean)/xscale)-RooMath::erf((x.min(rangeName)-mean)/xscale));
//     cout << "Int_gauss_dx(mean=" << mean << ",sigma=" << sigma << ", xmin=" << x.min(rangeName) << ", xmax=" << x.max(rangeName) << ")=" << ret << endl ;
  } else if(code==2) {
    ret = rootPiBy2*sigma*(RooMath::erf((mean.max(rangeName)-x)/xscale)-RooMath::erf((mean.min(rangeName)-x)/xscale));
  } else{
    cout << "Error in RooGaussian::analyticalIntegral" << endl;
  }
  return ret ;

}




//_____________________________________________________________________________
Int_t RooGaussian::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  if (matchArgs(directVars,generateVars,mean)) return 2 ;  
  return 0 ;
}



//_____________________________________________________________________________
void RooGaussian::generateEvent(Int_t code)
{
  assert(code==1 || code==2) ;
  Double_t xgen ;
  if(code==1){
    while(1) {    
      xgen = RooRandom::randomGenerator()->Gaus(mean,sigma);
      if (xgen<x.max() && xgen>x.min()) {
	x = xgen ;
	break;
      }
    }
  } else if(code==2){
    while(1) {    
      xgen = RooRandom::randomGenerator()->Gaus(x,sigma);
      if (xgen<mean.max() && xgen>mean.min()) {
	mean = xgen ;
	break;
      }
    }
  } else {
    cout << "error in RooGaussian generateEvent"<< endl;
  }

  return;
}

