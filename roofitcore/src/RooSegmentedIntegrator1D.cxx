/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator1D.cc,v 1.1 2003/05/09 20:48:23 wverkerke Exp $
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

}


RooSegmentedIntegrator1D::~RooSegmentedIntegrator1D()
{
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
    cout << "RooSegmentedIntegrator1D::checkLimits: bad range with min >= max" << endl;
    cout << "_xmax = " << _xmax << " _xmin = " << _xmin << endl ;
    cout << "_useIL = " << (_useIntegrandLimits?"T":"F") << endl ;
    return kFALSE;
  }
  return (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? kFALSE : kTRUE;
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

