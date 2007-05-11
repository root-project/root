/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooImproperIntegrator1D.cc,v 1.18 2006/07/03 15:37:11 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// Implementation of the abstract RooAbsIntegrator interface that can handle
// integration limits of +/-Infinity.


#include "RooFit.h"

#include "RooImproperIntegrator1D.h"
#include "RooImproperIntegrator1D.h"
#include "RooIntegrator1D.h"
#include "RooInvTransform.h"
#include "RooNumber.h"
#include "RooNumIntFactory.h"
#include "RooArgSet.h"

#include "Riostream.h"
#include <math.h>
#include "TClass.h"

ClassImp(RooImproperIntegrator1D)
;

// Register this class with RooNumIntConfig
void RooImproperIntegrator1D::registerIntegrator(RooNumIntFactory& fact)
{
  RooImproperIntegrator1D* proto = new RooImproperIntegrator1D() ;
  fact.storeProtoIntegrator(proto,RooArgSet(),RooIntegrator1D::Class()->GetName()) ;
  RooNumIntConfig::defaultConfig().method1DOpen().setLabel(proto->IsA()->GetName()) ;
}


RooImproperIntegrator1D::RooImproperIntegrator1D()
{
}

RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function) :
  RooAbsIntegrator(function),
  _useIntegrandLimits(kTRUE),
  _origFunc((RooAbsFunc*)&function),
  _function(0),
  _integrator1(0),
  _integrator2(0),
  _integrator3(0)
{
  initialize(&function) ;
}


RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooAbsIntegrator(function),
  _useIntegrandLimits(kTRUE),
  _origFunc((RooAbsFunc*)&function),
  _function(0),
  _config(config),
  _integrator1(0),
  _integrator2(0),
  _integrator3(0)
{
  initialize(&function) ;
}


RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function, Double_t xmin, Double_t xmax, const RooNumIntConfig& config) :
  RooAbsIntegrator(function),
  _xmin(xmin),
  _xmax(xmax),
  _useIntegrandLimits(kFALSE),
  _origFunc((RooAbsFunc*)&function),
  _function(0),
  _config(config),
  _integrator1(0),
  _integrator2(0),
  _integrator3(0)
{
  initialize(&function) ;
}


RooAbsIntegrator* RooImproperIntegrator1D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  return new RooImproperIntegrator1D(function,config) ;
}


void RooImproperIntegrator1D::initialize(const RooAbsFunc* function)
{
  if(!isValid()) {
    cout << "RooImproperIntegrator: cannot integrate invalid function" << endl;
    return;
  }
  // Create a new function object that uses the change of vars: x -> 1/x
  if (function) {
    _function= new RooInvTransform(*function);
  } else {
    function = _origFunc ;
    if (_integrator1) {
      delete _integrator1 ; 
      _integrator1 = 0 ;
    }
    if (_integrator2) {
      delete _integrator2 ; 
      _integrator2 = 0 ;
    }
    if (_integrator3) {
      delete _integrator3 ; 
      _integrator3 = 0 ;
    }
  }

  // partition the integration range into subranges that can each be
  // handled by RooIntegrator1D
  switch(_case= limitsCase()) {
  case ClosedBothEnds:
    // both limits are finite: use the plain trapezoid integrator
    _integrator1= new RooIntegrator1D(*function,_xmin,_xmax,_config);
    break;
  case OpenBothEnds:
    // both limits are infinite: integrate over (-1,+1) using
    // the plain trapezoid integrator...
    _integrator1= new RooIntegrator1D(*function,-1,+1,RooIntegrator1D::Trapezoid);
    // ...and integrate the infinite tails using the midpoint integrator
    _integrator2= new RooIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator3= new RooIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
    break;
  case OpenBelowSpansZero:
    // xmax >= 0 so integrate from (-inf,-1) and (-1,xmax)
    _integrator1= new RooIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator2= new RooIntegrator1D(*function,-1,_xmax,RooIntegrator1D::Trapezoid);
    break;
  case OpenBelow:
    // xmax < 0 so integrate from (-inf,xmax)
    _integrator1= new RooIntegrator1D(*_function,1/_xmax,0,RooIntegrator1D::Midpoint);
    break;
  case OpenAboveSpansZero:
    // xmin <= 0 so integrate from (xmin,+1) and (+1,+inf)
    _integrator1= new RooIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
    _integrator2= new RooIntegrator1D(*function,_xmin,+1,RooIntegrator1D::Trapezoid);
    break;
  case OpenAbove:
    // xmin > 0 so integrate from (xmin,+inf)
    _integrator1= new RooIntegrator1D(*_function,0,1/_xmin,RooIntegrator1D::Midpoint);
    break;
  case Invalid:
  default:
    _valid= kFALSE;
  }
}

RooImproperIntegrator1D::~RooImproperIntegrator1D() {
  if(0 != _integrator1) delete _integrator1;
  if(0 != _integrator2) delete _integrator2;
  if(0 != _integrator3) delete _integrator3;
  if(0 != _function) delete _function;
}

Bool_t RooImproperIntegrator1D::setLimits(Double_t xmin, Double_t xmax) {
  // Change our integration limits. Return kTRUE if the new limits are
  // ok, or otherwise kFALSE. Always returns kFALSE and does nothing
  // if this object was constructed to always use our integrand's limits.

  if(_useIntegrandLimits) {
    cout << "RooIntegrator1D::setLimits: cannot override integrand's limits" << endl;
    return kFALSE;
  }

  _xmin= xmin;
  _xmax= xmax;
  return checkLimits();
}

Bool_t RooImproperIntegrator1D::checkLimits() const {
  // Analyze the current limits to see if the same case applies.

  // Has either limit changed?
  if (_useIntegrandLimits) {
    if(_xmin == integrand()->getMinLimit(0) &&
       _xmax == integrand()->getMaxLimit(0)) return kTRUE;
  }

  // The limits have changed: can we use the same strategy?
  if(limitsCase() != _case) {
    // Reinitialize embedded integrators, will automatically propagate new limits
    const_cast<RooImproperIntegrator1D*>(this)->initialize() ;
    return kTRUE ;
  }

  // Reuse our existing integrators by updating their limits
  switch(_case) {
  case ClosedBothEnds:
    _integrator1->setLimits(_xmin,_xmax);
    break;
  case OpenBothEnds:
    // nothing has changed
    break;
  case OpenBelowSpansZero:
    _integrator2->setLimits(-1,_xmax);
    break;
  case OpenBelow:
    _integrator1->setLimits(1/_xmax,0);
    break;
  case OpenAboveSpansZero:
    _integrator2->setLimits(_xmin,+1);
    break;
  case OpenAbove:
    _integrator1->setLimits(0,1/_xmin);
    break;
  case Invalid:
  default:
    return kFALSE;
  }
  return kTRUE;
}

RooImproperIntegrator1D::LimitsCase RooImproperIntegrator1D::limitsCase() const {
  // Analyze the specified limits to determine which case applies.

  if(0 == integrand() || !integrand()->isValid()) return Invalid;

  if (_useIntegrandLimits) {
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }

  Bool_t inf1= RooNumber::isInfinite(_xmin);
  Bool_t inf2= RooNumber::isInfinite(_xmax);
  if(!inf1 && !inf2) {
    // both limits are finite
    return ClosedBothEnds;
  }
  else if(inf1 && inf2) {
    // both limits are infinite
    return OpenBothEnds;
  }
  else if(inf1) { // inf2==false
    if(_xmax >= 0) {
      return OpenBelowSpansZero;
    }
    else {
      return OpenBelow;
    }
  }
  else { // inf1==false && inf2==true
    if(_xmin <= 0) {
      return OpenAboveSpansZero;
    }
    else {
      return OpenAbove;
    }
  }
  // return Invalid; OSF-CC: Statement unreachable
}

Double_t RooImproperIntegrator1D::integral(const Double_t* yvec) {

  Double_t result(0);
  if(0 != _integrator1) result+= _integrator1->integral(yvec);
  if(0 != _integrator2) result+= _integrator2->integral(yvec);
  if(0 != _integrator3) result+= _integrator3->integral(yvec);
  return result;  
}
