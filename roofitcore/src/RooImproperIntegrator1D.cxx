/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooImproperIntegrator1D.cc,v 1.3 2001/09/15 00:26:02 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// Implementation of the abstract RooAbsIntegrator interface that can handle
// integration limits of +/-Infinity.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooImproperIntegrator1D.hh"
#include "RooFitCore/RooIntegrator1D.hh"
#include "RooFitCore/RooInvTransform.hh"
#include "RooFitCore/RooNumber.hh"

#include <iostream.h>
#include <math.h>

ClassImp(RooImproperIntegrator1D)
;

static const char rcsid[] =
"$Id: RooImproperIntegrator1D.cc,v 1.3 2001/09/15 00:26:02 david Exp $";

RooImproperIntegrator1D::RooImproperIntegrator1D(const RooAbsFunc& function) :
  RooAbsIntegrator(function),_function(0),_integrator1(0),_integrator2(0),_integrator3(0)
{
  if(!isValid()) {
    cout << "RooImproperIntegrator: cannot integrate invalid function" << endl;
    return;
  }
  // Create a new function object that uses the change of vars: x -> 1/x
  _function= new RooInvTransform(function);
  // partition the integration range into subranges that can each be
  // handled by RooIntegrator1D
  switch(_case= limitsCase()) {
  case ClosedBothEnds:
    // both limits are finite: use the plain trapezoid integrator
    _integrator1= new RooIntegrator1D(function,_xmin,_xmax,RooIntegrator1D::Trapezoid);
    break;
  case OpenBothEnds:
    // both limits are infinite: integrate over (-1,+1) using
    // the plain trapezoid integrator...
    _integrator1= new RooIntegrator1D(function,-1,+1,RooIntegrator1D::Trapezoid);
    // ...and integrate the infinite tails using the midpoint integrator
    _integrator2= new RooIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator3= new RooIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
    break;
  case OpenBelowSpansZero:
    // xmax >= 0 so integrate from (-inf,-1) and (-1,xmax)
    _integrator1= new RooIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator2= new RooIntegrator1D(function,-1,_xmax,RooIntegrator1D::Trapezoid);
    break;
  case OpenBelow:
    // xmax < 0 so integrate from (-inf,xmax)
    _integrator1= new RooIntegrator1D(*_function,1/_xmax,0,RooIntegrator1D::Midpoint);
    break;
  case OpenAboveSpansZero:
    // xmin <= 0 so integrate from (xmin,+1) and (+1,+inf)
    _integrator1= new RooIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
    _integrator2= new RooIntegrator1D(function,_xmin,+1,RooIntegrator1D::Trapezoid);
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

Bool_t RooImproperIntegrator1D::checkLimits() const {
  // Analyze the current limits to see if the same case applies.

  // Has either limit changed?
  if(_xmin == integrand()->getMinLimit(0) &&
     _xmax == integrand()->getMaxLimit(0)) return kTRUE;

  // The limits have changed: can we use the same strategy?
  if(limitsCase() != _case) return kFALSE;

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

  _xmin= integrand()->getMinLimit(0);
  _xmax= integrand()->getMaxLimit(0);
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

Double_t RooImproperIntegrator1D::integral() {
  Double_t result(0);
  if(0 != _integrator1) result+= _integrator1->integral();
  if(0 != _integrator2) result+= _integrator2->integral();
  if(0 != _integrator3) result+= _integrator3->integral();
  return result;  
}
