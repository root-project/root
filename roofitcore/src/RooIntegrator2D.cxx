/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
// RooIntegrator2D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFitCore/RooIntegrator2D.hh"
#include "RooFitCore/RooIntegrator1DBinding.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooIntegratorConfig.hh"

#include <assert.h>

ClassImp(RooIntegrator2D)
;

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, RooIntegrator1D::SummationRule rule,
				 Int_t maxSteps, Double_t eps) : 
  RooIntegrator1D(*(_xint=new RooIntegrator1DBinding(*(_xIntegrator=new RooIntegrator1D(function,rule,maxSteps,eps)))),rule,maxSteps,eps)
{
} 

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, const RooIntegratorConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegrator1DBinding(*(_xIntegrator=new RooIntegrator1D(function,config)))),config)
{
} 


RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 SummationRule rule, Int_t maxSteps, Double_t eps) : 
  RooIntegrator1D(*(_xint=new RooIntegrator1DBinding(*(_xIntegrator=new RooIntegrator1D(function,ymin,ymax,rule,maxSteps,eps)))),xmin,xmax,rule,maxSteps,eps)
{
} 

RooIntegrator2D::RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 const RooIntegratorConfig& config) :
  RooIntegrator1D(*(_xint=new RooIntegrator1DBinding(*(_xIntegrator=new RooIntegrator1D(function,ymin,ymax,config)))),xmin,xmax,config)
{
} 


RooIntegrator2D::~RooIntegrator2D() 
{
  delete _xint ;
  delete _xIntegrator ;
}

