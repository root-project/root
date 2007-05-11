/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooIntegrator2D.cc,v 1.13 2006/07/03 15:37:11 wverkerke Exp $
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
// RooIntegrator2D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFit.h"

#include "TClass.h"
#include "RooIntegrator2D.h"
#include "RooArgSet.h"
#include "RooIntegratorBinding.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooNumIntFactory.h"

#include <assert.h>

ClassImp(RooIntegrator2D)
;

// Register this class with RooNumIntConfig
void RooIntegrator2D::registerIntegrator(RooNumIntFactory& fact)
{
  RooIntegrator2D* proto = new RooIntegrator2D() ;
  fact.storeProtoIntegrator(proto,RooArgSet(),RooIntegrator1D::Class()->GetName()) ;
  RooNumIntConfig::defaultConfig().method2D().setLabel(proto->IsA()->GetName()) ;
}

RooIntegrator2D::RooIntegrator2D()
{
}

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, RooIntegrator1D::SummationRule rule,
				 Int_t maxSteps, Double_t eps) : 
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,rule,maxSteps,eps)))),rule,maxSteps,eps)
{
} 

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 SummationRule rule, Int_t maxSteps, Double_t eps) : 
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,ymin,ymax,rule,maxSteps,eps)))),xmin,xmax,rule,maxSteps,eps)
{
} 

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,config)))),config)
{
} 


RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 const RooNumIntConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooIntegrator1D(function,ymin,ymax,config)))),xmin,xmax,config)
{
} 

RooAbsIntegrator* RooIntegrator2D::clone(const RooAbsFunc& function, const RooNumIntConfig& config) const
{
  return new RooIntegrator2D(function,config) ;
}


RooIntegrator2D::~RooIntegrator2D() 
{
  delete _xint ;
  delete _xIntegrator ;
}

Bool_t RooIntegrator2D::checkLimits() const 
{
  Bool_t ret = RooIntegrator1D::checkLimits() ;
  ret &= _xIntegrator->checkLimits() ;
  return ret ;
}
