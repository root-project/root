/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooImproperIntegrator1D.cc,v 1.1 2001/08/08 23:11:24 david Exp $
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
"$Id: RooImproperIntegrator1D.cc,v 1.1 2001/08/08 23:11:24 david Exp $";

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
  Double_t xmin(function.getMinLimit(0)), xmax(function.getMaxLimit(0));
  Bool_t inf1= RooNumber::isInfinite(xmin);
  Bool_t inf2= RooNumber::isInfinite(xmax);
  if(!inf1 && !inf2) {
    // both limits are finite: use the plain trapezoid integrator
    _integrator1= new RooIntegrator1D(function,RooIntegrator1D::Trapezoid);
  }
  else if(inf1 && inf2) {
    Double_t chopAt(1);
    // both limits are infinite: integrate over (-1,+1) using
    // the plain trapezoid integrator...
    _integrator1= new RooIntegrator1D(function,-1,+1,RooIntegrator1D::Trapezoid);
    // ...and integrate the infinite tails using the midpoint integrator
    _integrator2= new RooIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
    _integrator3= new RooIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
  }
  else if(inf1) { // inf2==false
    if(xmax >= 0) {
      _integrator1= new RooIntegrator1D(*_function,-1,0,RooIntegrator1D::Midpoint);
      _integrator2= new RooIntegrator1D(function,-1,xmax,RooIntegrator1D::Trapezoid);
    }
    else {
      _integrator1= new RooIntegrator1D(*_function,1/xmax,0,RooIntegrator1D::Midpoint);
    }
  }
  else { // inf1==false && inf2==true
    if(xmin <= 0) {
      _integrator1= new RooIntegrator1D(*_function,0,+1,RooIntegrator1D::Midpoint);
      _integrator2= new RooIntegrator1D(function,xmin,+1,RooIntegrator1D::Trapezoid);
    }
    else {
      _integrator1= new RooIntegrator1D(*_function,0,1/xmin,RooIntegrator1D::Midpoint);
    }
  }
}

RooImproperIntegrator1D::~RooImproperIntegrator1D() {
  if(0 != _function) delete _function;
  if(0 != _integrator1) delete _integrator1;
  if(0 != _integrator2) delete _integrator2;
  if(0 != _integrator3) delete _integrator3;
}

Bool_t RooImproperIntegrator1D::checkLimits() const {
  return kFALSE;
}

Double_t RooImproperIntegrator1D::integral() {
  Double_t result(0);
  if(0 != _integrator1) result+= _integrator1->integral();
  if(0 != _integrator2) result+= _integrator2->integral();
  if(0 != _integrator3) result+= _integrator3->integral();
  return result;  
}
