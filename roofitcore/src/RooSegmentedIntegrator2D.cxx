/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator2D.cc,v 1.1 2003/05/07 21:06:25 wverkerke Exp $
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
// RooSegmentedIntegrator2D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFitCore/RooSegmentedIntegrator2D.hh"
#include "RooFitCore/RooIntegratorBinding.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooIntegratorConfig.hh"

#include <assert.h>

ClassImp(RooSegmentedIntegrator2D)
;

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, RooIntegrator1D::SummationRule rule,
				 Int_t maxSteps, Double_t eps) : 
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,nseg,rule,maxSteps,eps)))),nseg,rule,maxSteps,eps)
{
} 

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, const RooIntegratorConfig& config) :
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,nseg,config)))),nseg,config)
{
} 


RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 RooIntegrator1D::SummationRule rule, Int_t maxSteps, Double_t eps) : 
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,nseg,ymin,ymax,rule,maxSteps,eps)))),nseg,xmin,xmax,rule,maxSteps,eps)
{
} 

RooSegmentedIntegrator2D::RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, Double_t xmin, Double_t xmax,
				 Double_t ymin, Double_t ymax,
				 const RooIntegratorConfig& config) :
  RooSegmentedIntegrator1D(*(_xint=new RooIntegratorBinding(*(_xIntegrator=new RooSegmentedIntegrator1D(function,nseg,ymin,ymax,config)))),nseg,xmin,xmax,config)
{
} 


RooSegmentedIntegrator2D::~RooSegmentedIntegrator2D() 
{
  delete _xint ;
  delete _xIntegrator ;
}

