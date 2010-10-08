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
// RooSegmentedIntegrator1D implements an adaptive one-dimensional 
// numerical integration algorithm.
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"
#include "RooSegmentedIntegrator1D.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooMsgService.h"
#include "RooNumIntFactory.h"

#include <assert.h>



ClassImp(RooSegmentedIntegrator1D)
;

// Register this class with RooNumIntConfig

//_____________________________________________________________________________
void RooSegmentedIntegrator1D::registerIntegrator(RooNumIntFactory& fact)
{
  // Register RooSegmentedIntegrator1D, its parameters, dependencies and capabilities with RooNumIntFactory

  RooRealVar numSeg("numSeg","Number of segments",3) ;
  fact.storeProtoIntegrator(new RooSegmentedIntegrator1D(),numSeg,RooIntegrator1D::Class()->GetName()) ;
}
 


//_____________________________________________________________________________
RooSegmentedIntegrator1D::RooSegmentedIntegrator1D()
{
  // Constructor
  //
  // coverity[UNINIT_CTOR]
}



//_____________________________________________________________________________
RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooAbsIntegrator(function), _config(config)
{
  // Constructor of integral on given function binding and with given configuration. The
  // integration limits are taken from the definition in the function binding

  _nseg = (Int_t) config.getConfigSection(IsA()->GetName()).getRealValue("numSeg",3) ;
  _useIntegrandLimits= kTRUE;

  _valid= initialize();
} 



//_____________________________________________________________________________
RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
						   const RooNumIntConfig& config) :
  RooAbsIntegrator(function), _config(config) 
{
  // Constructor integral on given function binding, with given configuration and
  // explicit definition of integration range

  _nseg = (Int_t) config.getConfigSection(IsA()->GetName()).getRealValue("numSeg",3) ;
  _useIntegrandLimits= kFALSE;
  _xmin= xmin;
  _xmax= xmax;

  _valid= initialize();
} 



//_____________________________________________________________________________
RooAbsIntegrator* RooSegmentedIntegrator1D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  // Virtual constructor with given function and configuration. Needed by RooNumIntFactory
  
  return new RooSegmentedIntegrator1D(function,config) ;
}



typedef RooIntegrator1D* pRooIntegrator1D ;

//_____________________________________________________________________________
Bool_t RooSegmentedIntegrator1D::initialize()
{
  // One-time integrator initialization

  _array = 0 ;
  
  Bool_t limitsOK = checkLimits(); 
  if (!limitsOK) return kFALSE ;

  // Make array of integrators for each segment
  _array = new pRooIntegrator1D[_nseg] ;

  Int_t i ;

  Double_t segSize = (_xmax - _xmin) / _nseg ;

  // Adjust integrator configurations for reduced intervals
  _config.setEpsRel(_config.epsRel()/sqrt(1.*_nseg)) ;
  _config.setEpsAbs(_config.epsAbs()/sqrt(1.*_nseg)) ;
    
  for (i=0 ; i<_nseg ; i++) {
    _array[i] = new RooIntegrator1D(*_function,_xmin+i*segSize,_xmin+(i+1)*segSize,_config) ;
  }

  return kTRUE ;
}



//_____________________________________________________________________________
RooSegmentedIntegrator1D::~RooSegmentedIntegrator1D()
{
  // Destructor
}



//_____________________________________________________________________________
Bool_t RooSegmentedIntegrator1D::setLimits(Double_t* xmin, Double_t* xmax) 
{
  // Change our integration limits. Return kTRUE if the new limits are
  // ok, or otherwise kFALSE. Always returns kFALSE and does nothing
  // if this object was constructed to always use our integrand's limits.

  if(_useIntegrandLimits) {
    oocoutE((TObject*)0,InputArguments) << "RooSegmentedIntegrator1D::setLimits: cannot override integrand's limits" << endl;
    return kFALSE;
  }
  _xmin= *xmin;
  _xmax= *xmax;
  return checkLimits();
}



//_____________________________________________________________________________
Bool_t RooSegmentedIntegrator1D::checkLimits() const 
{
  // Check that our integration range is finite and otherwise return kFALSE.
  // Update the limits from the integrand if requested.
  
  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  if(_range <= 0) {
    oocoutE((TObject*)0,InputArguments) << "RooIntegrator1D::checkLimits: bad range with min >= max" << endl;
    return kFALSE;
  }
  Bool_t ret =  (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? kFALSE : kTRUE;

  // Adjust component integrators, if already created
  if (_array && ret) {
    Double_t segSize = (_xmax - _xmin) / _nseg ;
    Int_t i ;
    for (i=0 ; i<_nseg ; i++) {
      _array[i]->setLimits(_xmin+i*segSize,_xmin+(i+1)*segSize) ;
    }
  }

  return ret ;
}




//_____________________________________________________________________________
Double_t RooSegmentedIntegrator1D::integral(const Double_t *yvec) 
{
  // Evaluate integral at given function binding parameter values

  assert(isValid());

  Int_t i ;
  Double_t result(0) ;
  for (i=0 ; i<_nseg ; i++) {
    result += _array[i]->integral(yvec) ;
  }

  return result;
}

