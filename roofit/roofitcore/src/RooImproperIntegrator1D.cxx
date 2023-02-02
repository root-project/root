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

/**
\file RooImproperIntegrator1D.cxx
\class RooImproperIntegrator1D
\ingroup Roofitcore

Special numeric integrator that can handle integrals over open domains.
To this end the range is cut in up three pieces: [-inf,-1],[-1,+1] and [+1,inf]
and the outer two pieces, if required are calculated using a 1/x transform
**/

#include "RooImproperIntegrator1D.h"
#include "RooIntegrator1D.h"
#include "RooInvTransform.h"
#include "RooNumber.h"
#include "RooNumIntFactory.h"
#include "RooArgSet.h"
#include "RooMsgService.h"

#include "Riostream.h"
#include <math.h>
#include "TClass.h"


ClassImp(RooImproperIntegrator1D);

// Register this class with RooNumIntConfig

////////////////////////////////////////////////////////////////////////////////
/// Register RooImproperIntegrator1D, its parameters and capabilities with RooNumIntFactory

void RooImproperIntegrator1D::registerIntegrator(RooNumIntFactory& fact)
{
  RooImproperIntegrator1D* proto = new RooImproperIntegrator1D() ;
  fact.storeProtoIntegrator(proto,RooArgSet(),RooIntegrator1D::Class()->GetName()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooImproperIntegrator1D::RooImproperIntegrator1D() :
  _case(ClosedBothEnds), _xmin(-10), _xmax(10), _useIntegrandLimits(true)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with function binding. The integration range is taken from the
/// definition in the function binding

RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function) :
  RooAbsIntegrator(function),
  _useIntegrandLimits(true),
  _origFunc((RooAbsFunc*)&function)
{
  initialize(&function) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with function binding and configuration object. The integration range is taken
/// from the definition in the function binding

RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooAbsIntegrator(function),
  _useIntegrandLimits(true),
  _origFunc((RooAbsFunc*)&function),
  _config(config)
{
  initialize(&function) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with function binding, definition of integration range and configuration object

RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function, double xmin, double xmax, const RooNumIntConfig& config) :
  RooAbsIntegrator(function),
  _xmin(xmin),
  _xmax(xmax),
  _useIntegrandLimits(false),
  _origFunc((RooAbsFunc*)&function),
  _config(config)
{
  initialize(&function) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return clone of integrator with given function and configuration. Needed by RooNumIntFactory.

RooAbsIntegrator* RooImproperIntegrator1D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  return new RooImproperIntegrator1D(function,config) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize the integrator, construct and initialize subintegrators

void RooImproperIntegrator1D::initialize(const RooAbsFunc* function)
{
  if(!isValid()) {
    oocoutE(nullptr,Integration) << "RooImproperIntegrator: cannot integrate invalid function" << std::endl;
    return;
  }
  // Create a new function object that uses the change of vars: x -> 1/x
  if (function) {
    _function= std::make_unique<RooInvTransform>(*function);
  } else {
    function = _origFunc ;
    _integrator1.reset();
    _integrator2.reset();
    _integrator3.reset();
  }

  // Helper function to create a new configuration that is just like the one
  // associated to this integrator, but with a different summation rule.
  auto makeIntegrator1D = [&](RooAbsFunc const& func,
                              double xmin, double xmax,
                              RooIntegrator1D::SummationRule rule) {
      RooNumIntConfig newConfig{_config}; // copy default configuration
      newConfig.getConfigSection("RooIntegrator1D").setCatIndex("sumRule", rule);
      return std::make_unique<RooIntegrator1D>(func, xmin, xmax, newConfig);
  };

  // partition the integration range into subranges that can each be
  // handled by RooIntegrator1D
  switch(_case= limitsCase()) {
  case ClosedBothEnds:
    // both limits are finite: use the plain trapezoid integrator
    _integrator1 = std::make_unique<RooIntegrator1D>(*function,_xmin,_xmax,_config);
    break;
  case OpenBothEnds:
    // both limits are infinite: integrate over (-1,+1) using
    // the plain trapezoid integrator...
    _integrator1 = makeIntegrator1D(*function,-1,+1,RooIntegrator1D::Trapezoid);
    // ...and integrate the infinite tails using the midpoint integrator
    _integrator2 = makeIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator3 = makeIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
    break;
  case OpenBelowSpansZero:
    // xmax >= 0 so integrate from (-inf,-1) and (-1,xmax)
    _integrator1 = makeIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator2 = makeIntegrator1D(*function,-1,_xmax,RooIntegrator1D::Trapezoid);
    break;
  case OpenBelow:
    // xmax < 0 so integrate from (-inf,xmax)
    _integrator1 = makeIntegrator1D(*_function,1/_xmax,0,RooIntegrator1D::Midpoint);
    break;
  case OpenAboveSpansZero:
    // xmin <= 0 so integrate from (xmin,+1) and (+1,+inf)
    _integrator1 = makeIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
    _integrator2 = makeIntegrator1D(*function,_xmin,+1,RooIntegrator1D::Trapezoid);
    break;
  case OpenAbove:
    // xmin > 0 so integrate from (xmin,+inf)
    _integrator1 = makeIntegrator1D(*_function,0,1/_xmin,RooIntegrator1D::Midpoint);
    break;
  case Invalid:
  default:
    _valid= false;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Change our integration limits. Return true if the new limits are
/// ok, or otherwise false. Always returns false and does nothing
/// if this object was constructed to always use our integrand's limits.

bool RooImproperIntegrator1D::setLimits(double *xmin, double *xmax)
{
  if(_useIntegrandLimits) {
    oocoutE(nullptr,Integration) << "RooIntegrator1D::setLimits: cannot override integrand's limits" << std::endl;
    return false;
  }

  _xmin= *xmin;
  _xmax= *xmax;
  return checkLimits();
}


////////////////////////////////////////////////////////////////////////////////
/// Check if the limits are valid. For this integrator all limit configurations
/// are valid, but if the limits change between two calculate() calls it
/// may be necessary to reconfigure (e.g. if an open ended range becomes
/// a closed range

bool RooImproperIntegrator1D::checkLimits() const
{
  // Has either limit changed?
  if (_useIntegrandLimits) {
    if(_xmin == integrand()->getMinLimit(0) &&
       _xmax == integrand()->getMaxLimit(0)) return true;
  }

  // The limits have changed: can we use the same strategy?
  if(limitsCase() != _case) {
    // Reinitialize embedded integrators, will automatically propagate new limits
    const_cast<RooImproperIntegrator1D*>(this)->initialize() ;
    return true ;
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
    return false;
  }
  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Classify the type of limits we have: OpenBothEnds,ClosedBothEnds,OpenBelow or OpenAbove.

RooImproperIntegrator1D::LimitsCase RooImproperIntegrator1D::limitsCase() const
{
  // Analyze the specified limits to determine which case applies.
  if(0 == integrand() || !integrand()->isValid()) return Invalid;

  if (_useIntegrandLimits) {
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }

  bool inf1= RooNumber::isInfinite(_xmin);
  bool inf2= RooNumber::isInfinite(_xmax);
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


////////////////////////////////////////////////////////////////////////////////
/// Calculate the integral at the given parameter values of the function binding

double RooImproperIntegrator1D::integral(const double* yvec)
{
  double result(0);
  if(_integrator1) result+= _integrator1->integral(yvec);
  if(_integrator2) result+= _integrator2->integral(yvec);
  if(_integrator3) result+= _integrator3->integral(yvec);
  return result;
}
