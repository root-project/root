/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator1D.rdl,v 1.1 2003/05/09 20:48:23 wverkerke Exp $
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
#ifndef ROO_SEGMENTED_INTEGRATOR_1D
#define ROO_SEGMENTED_INTEGRATOR_1D

#include "RooFitCore/RooAbsIntegrator.hh"
#include "RooFitCore/RooIntegrator1D.hh"
#include "RooFitCore/RooIntegratorConfig.hh" 

class RooSegmentedIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, 
			   RooIntegrator1D::SummationRule rule=RooIntegrator1D::Trapezoid,
			   Int_t maxSteps= 0, Double_t eps= 0) ; 

  RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, const RooIntegratorConfig& config) ;

  RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, Double_t xmin, Double_t xmax,
			   RooIntegrator1D::SummationRule rule= RooIntegrator1D::Trapezoid, 
			   Int_t maxSteps= 0, Double_t eps= 0) ; 

  RooSegmentedIntegrator1D(const RooAbsFunc& function, Int_t nSegments, Double_t xmin, Double_t xmax, 
			   const RooIntegratorConfig& config) ;
  virtual ~RooSegmentedIntegrator1D();

  virtual Bool_t checkLimits() const;
  virtual Double_t integral(const Double_t *yvec=0) ;
  Bool_t setLimits(Double_t xmin, Double_t xmax);

protected:

  mutable Double_t _xmin ;
  mutable Double_t _xmax ;
  mutable Double_t _range ;
  Bool_t _valid ;
  Int_t _nseg ; // Number of segments 
  Bool_t _useIntegrandLimits ;

  RooIntegratorConfig _config ;  
  RooIntegrator1D** _array ; // Array of segment integrators

  Bool_t initialize();

  ClassDef(RooSegmentedIntegrator1D,0) // 1-dimensional piece-wise numerical integration engine
};

#endif
