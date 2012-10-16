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
// RooIntegrator2D implements a numeric two-dimensiona integrator
// in terms of a recursive application of RooIntegrator1D
// END_HTML
//


#include "RooFit.h"

#include "TClass.h"
#include "RooIntegrator2D.h"
#include "RooArgSet.h"
#include "RooIntegratorBinding.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooNumIntFactory.h"

#include <assert.h>

using namespace std;

ClassImp(RooIntegrator2D)
;


//_____________________________________________________________________________
void RooIntegrator2D::registerIntegrator(RooNumIntFactory& fact)
{
  // Register RooIntegrator2D, is parameters and capabilities with RooNumIntFactory

  RooIntegrator2D* proto = new RooIntegrator2D() ;
  fact.storeProtoIntegrator(proto,RooArgSet(),RooIntegrator1D::Class()->GetName()) ;
  RooNumIntConfig::defaultConfig().method2D().setLabel(proto->IsA()->GetName()) ;
}


//_____________________________________________________________________________
RooIntegrator2D::RooIntegrator2D() :
  _xIntegrator(0), _xint(0)
{
  // Default constructor

}


//_____________________________________________________________________________
RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, RooIntegrator1D::SummationRule rule,
				 Int_t maxSteps, Double_t eps) : 
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,rule,maxSteps,eps)))),rule,maxSteps,eps)
{
  // Constructor with a given function binding, summation rule,
  // maximum number of steps and conversion tolerance. The integration
  // limits are taken from the definition in the function binding.
} 


//_____________________________________________________________________________
RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 SummationRule rule, Int_t maxSteps, Double_t eps) : 
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,ymin,ymax,rule,maxSteps,eps)))),xmin,xmax,rule,maxSteps,eps)
{
  // Constructor with a given function binding, summation rule,
  // maximum number of steps, conversion tolerance and an explicit
  // choice of integration limits on both dimensions.
} 


//_____________________________________________________________________________
RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,config)))),config)
{
  // Constructor with a function binding and a configuration object.
  // The integration limits are taken from the definition in the function
  // binding
} 



//_____________________________________________________________________________
RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 const RooNumIntConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,ymin,ymax,config)))),xmin,xmax,config)
{
  // Constructor with a function binding, a configuration object and
  // an explicit definition of the integration limits.
} 


//_____________________________________________________________________________
RooAbsIntegrator* RooIntegrator2D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  // Clone integrator with new function and configuration. Needed to support RooNumIntFactory
  return new RooIntegrator2D(function,config) ;
}



//_____________________________________________________________________________
RooIntegrator2D::~RooIntegrator2D() 
{
  // Destructor

  delete _xint ;
  delete _xIntegrator ;
}


//_____________________________________________________________________________
Bool_t RooIntegrator2D::checkLimits() const 
{
  // Verify that the limits are OK for this integrator (i.e. no open-ended ranges)

  Bool_t ret = RooIntegrator1D::checkLimits() ;
  ret &= _xIntegrator->checkLimits() ;
  return ret ;
}
