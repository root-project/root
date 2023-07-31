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
\file RooIntegrator2D.cxx
\class RooIntegrator2D
\ingroup Roofitcore

RooIntegrator2D implements a numeric two-dimensiona integrator
in terms of a recursive application of RooIntegrator1D
**/


#include "TClass.h"
#include "RooIntegrator2D.h"
#include "RooArgSet.h"
#include "RooIntegratorBinding.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooNumIntFactory.h"

#include <assert.h>

using namespace std;

ClassImp(RooIntegrator2D);


////////////////////////////////////////////////////////////////////////////////
/// Register RooIntegrator2D, is parameters and capabilities with RooNumIntFactory

void RooIntegrator2D::registerIntegrator(RooNumIntFactory& fact)
{
  auto creator = [](const RooAbsFunc& function, const RooNumIntConfig& config) {
    return std::make_unique<RooIntegrator2D>(function,config);
  };
  std::string name = "RooIntegrator2D";
  fact.registerPlugin(name, creator, {},
                    /*canIntegrate1D=*/false,
                    /*canIntegrate2D=*/true,
                    /*canIntegrateND=*/false,
                    /*canIntegrateOpenEnded=*/false,
                    /*depName=*/"RooIntegrator1D");
  RooNumIntConfig::defaultConfig().method2D().setLabel(name) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with a given function binding, summation rule,
/// maximum number of steps and conversion tolerance. The integration
/// limits are taken from the definition in the function binding.

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc &function, RooIntegrator1D::SummationRule rule, Int_t maxSteps,
                                 double eps)
   : RooIntegrator1D(
        *(_xint = new RooIntegratorBinding(std::make_unique<RooIntegrator1D>(function, rule, maxSteps, eps))), rule,
        maxSteps, eps)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with a given function binding, summation rule,
/// maximum number of steps, conversion tolerance and an explicit
/// choice of integration limits on both dimensions.

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, double xmin, double xmax,
             double ymin, double ymax,
             SummationRule rule, Int_t maxSteps, double eps) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(std::make_unique<RooIntegrator1D>(function,ymin,ymax,rule,maxSteps,eps))),xmin,xmax,rule,maxSteps,eps)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor with a function binding and a configuration object.
/// The integration limits are taken from the definition in the function
/// binding

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(std::make_unique<RooIntegrator1D>(function,config))),config)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with a function binding, a configuration object and
/// an explicit definition of the integration limits.

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, double xmin, double xmax,
             double ymin, double ymax,
             const RooNumIntConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(std::make_unique<RooIntegrator1D>(function,ymin,ymax,config))),xmin,xmax,config)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooIntegrator2D::~RooIntegrator2D()
{
  delete _xint ;
}


////////////////////////////////////////////////////////////////////////////////
/// Verify that the limits are OK for this integrator (i.e. no open-ended ranges)

bool RooIntegrator2D::checkLimits() const
{
  bool ret = RooIntegrator1D::checkLimits() ;
  ret &= _xint->integrator().checkLimits() ;
  return ret ;
}
