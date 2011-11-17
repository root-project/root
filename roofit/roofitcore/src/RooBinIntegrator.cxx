/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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
// RooBinIntegrator implements an adaptive one-dimensional 
// numerical integration algorithm. 
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"
#include "RooBinIntegrator.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooIntegratorBinding.h"
#include "RooNumIntConfig.h"
#include "RooNumIntFactory.h"
#include "RooMsgService.h"

#include <assert.h>



ClassImp(RooBinIntegrator)
;

// Register this class with RooNumIntConfig

//_____________________________________________________________________________
void RooBinIntegrator::registerIntegrator(RooNumIntFactory& fact)
{
  // Register RooBinIntegrator, is parameters and capabilities with RooNumIntFactory

  RooRealVar numBins("numBins","Number of bins in range",100) ;
  RooBinIntegrator* proto = new RooBinIntegrator() ;
  fact.storeProtoIntegrator(proto,RooArgSet(numBins)) ;
  RooNumIntConfig::defaultConfig().method1D().setLabel(proto->IsA()->GetName()) ;
}



//_____________________________________________________________________________
RooBinIntegrator::RooBinIntegrator() :  _x(0)
{
  // Default constructor
}


//_____________________________________________________________________________
RooBinIntegrator::RooBinIntegrator(const RooAbsFunc& function) : 
  RooAbsIntegrator(function)
{
  // Construct integrator on given function binding binding

  _useIntegrandLimits= kTRUE;
  assert(0 != integrand() && integrand()->isValid());

  // Allocate coordinate buffer size after number of function dimensions
  _x = new Double_t[_function->getDimension()] ;

  _xmin= integrand()->getMinLimit(0);
  _xmax= integrand()->getMaxLimit(0);
  checkLimits();

} 


//_____________________________________________________________________________
RooBinIntegrator::RooBinIntegrator(const RooAbsFunc& function, const RooNumIntConfig& config) : 
  RooAbsIntegrator(function)
{
  // Construct integrator on given function binding binding

  const RooArgSet& configSet = config.getConfigSection(IsA()->GetName()) ;  
  _numBins = (Int_t) configSet.getRealValue("numBins") ;
  assert(0 != integrand() && integrand()->isValid());

  // Allocate coordinate buffer size after number of function dimensions
  _x = new Double_t[_function->getDimension()] ;

  _xmin= integrand()->getMinLimit(0);
  _xmax= integrand()->getMaxLimit(0);
  checkLimits();

} 


//_____________________________________________________________________________
RooAbsIntegrator* RooBinIntegrator::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  // Clone integrator with new function binding and configuration. Needed by RooNumIntFactory
  return new RooBinIntegrator(function,config) ;
}





//_____________________________________________________________________________
RooBinIntegrator::~RooBinIntegrator()
{
  // Destructor
  if(_x) delete[] _x;

}


//_____________________________________________________________________________
Bool_t RooBinIntegrator::setLimits(Double_t *xmin, Double_t *xmax) 
{
  // Change our integration limits. Return kTRUE if the new limits are
  // ok, or otherwise kFALSE. Always returns kFALSE and does nothing
  // if this object was constructed to always use our integrand's limits.

  if(_useIntegrandLimits) {
    oocoutE((TObject*)0,Integration) << "RooBinIntegrator::setLimits: cannot override integrand's limits" << endl;
    return kFALSE;
  }
  _xmin= *xmin;
  _xmax= *xmax;
  return checkLimits();
}


//_____________________________________________________________________________
Bool_t RooBinIntegrator::checkLimits() const 
{
  // Check that our integration range is finite and otherwise return kFALSE.
  // Update the limits from the integrand if requested.

  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  _binWidth = _range/_numBins;
  if(_range < 0) {
    oocoutE((TObject*)0,Integration) << "RooBinIntegrator::checkLimits: bad range with min >= max (_xmin = " << _xmin << " _xmax = " << _xmax << ")" << endl;
    return kFALSE;
  }
  return (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? kFALSE : kTRUE;
}


//_____________________________________________________________________________
Double_t RooBinIntegrator::integral(const Double_t *yvec) 
{
  // Calculate numeric integral at given set of function binding parameters

  assert(isValid());

  double sum = 0., thisX = 0.;
  for(int i=0; i<_numBins; ++i){
    thisX = _xmin + (i+0.5)*_binWidth;
    sum+= integrand(xvec(thisX))*_binWidth;
  }
  //  cout <<" sum = " <<sum<<endl;
  return sum;

}


