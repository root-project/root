/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator1D.cc,v 1.3 2003/05/14 02:58:40 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooSegmentedIntegrator1D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFitCore/RooSegmentedIntegrator1D.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooIntegratorConfig.hh"

#include <assert.h>

ClassImp(RooSegmentedIntegrator1D)
;

RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, 
						   RooIntegrator1D::SummationRule rule,
						   Int_t maxSteps, Double_t eps) : 
  RooAbsIntegrator(function), _nseg(nSegments) 
{
  // Use this form of the constructor to integrate over the function's default range.
  _config.setSummationRule1D(rule) ;
  _config.setMaxSteps1D(maxSteps) ;
  _config.setEpsilonRel1D(eps) ;
  _config.setEpsilonAbs1D(eps) ;

  _useIntegrandLimits= kTRUE;
  _valid= initialize();
} 

RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, const RooIntegratorConfig& config) :
  RooAbsIntegrator(function), _nseg(nSegments), _config(config)
{
  // Use this form of the constructor to integrate over the function's default range.
  _useIntegrandLimits= kTRUE;

  _valid= initialize();
} 


RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, Double_t xmin, Double_t xmax,
						   RooIntegrator1D::SummationRule rule, 
						   Int_t maxSteps, Double_t eps) : 
  RooAbsIntegrator(function), _nseg(nSegments)
{
  // Use this form of the constructor to override the function's default range.
  _xmin= xmin;
  _xmax= xmax;

  _config.setSummationRule1D(rule) ;
  _config.setMaxSteps1D(maxSteps) ;
  _config.setEpsilonRel1D(eps) ;
  _config.setEpsilonAbs1D(eps) ;

  _useIntegrandLimits= kFALSE;
  _valid= initialize();
} 

RooSegmentedIntegrator1D::RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, Double_t xmin, Double_t xmax,
						   const RooIntegratorConfig& config) :
  RooAbsIntegrator(function), _nseg(nSegments), _config(config) 
{
  // Use this form of the constructor to override the function's default range.

  _useIntegrandLimits= kFALSE;
  _xmin= xmin;
  _xmax= xmax;

  _valid= initialize();
} 



typedef RooIntegrator1D* pRooIntegrator1D ;
Bool_t RooSegmentedIntegrator1D::initialize()
{
  _array = 0 ;
  
  Bool_t limitsOK = checkLimits(); 
  if (!limitsOK) return kFALSE ;

  // Make array of integrators for each segment
  _array = new pRooIntegrator1D[_nseg] ;

  Int_t i ;

  Double_t segSize = (_xmax - _xmin) / _nseg ;

  // Adjust integrator configurations for reduced intervals
  _config.setEpsilonRel1D(_config.epsilonRel1D()/sqrt(_nseg)) ;
  _config.setEpsilonAbs1D(_config.epsilonAbs1D()/sqrt(_nseg)) ;
    
  for (i=0 ; i<_nseg ; i++) {
    _array[i] = new RooIntegrator1D(*_function,_xmin+i*segSize,_xmin+(i+1)*segSize,_config) ;
  }

  return kTRUE ;
}


RooSegmentedIntegrator1D::~RooSegmentedIntegrator1D()
{
}


Bool_t RooSegmentedIntegrator1D::setLimits(Double_t xmin, Double_t xmax) {
  // Change our integration limits. Return kTRUE if the new limits are
  // ok, or otherwise kFALSE. Always returns kFALSE and does nothing
  // if this object was constructed to always use our integrand's limits.

  if(_useIntegrandLimits) {
    cout << "RooSegmentedIntegrator1D::setLimits: cannot override integrand's limits" << endl;
    return kFALSE;
  }
  _xmin= xmin;
  _xmax= xmax;
  return checkLimits();
}



Bool_t RooSegmentedIntegrator1D::checkLimits() const {
  // Check that our integration range is finite and otherwise return kFALSE.
  // Update the limits from the integrand if requested.

  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  if(_range <= 0) {
    cout << "RooIntegrator1D::checkLimits: bad range with min >= max" << endl;
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




Double_t RooSegmentedIntegrator1D::integral(const Double_t *yvec) 
{
  assert(isValid());

  Int_t i ;
  Double_t result(0) ;
  for (i=0 ; i<_nseg ; i++) {
    result += _array[i]->integral(yvec) ;
  }

  return result;
}

