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
#ifndef ROO_SEGMENTED_INTEGRATOR_2D
#define ROO_SEGMENTED_INTEGRATOR_2D

#include "RooFitCore/RooSegmentedIntegrator1D.hh"
#include "RooFitCore/RooIntegrator1D.hh"
class RooIntegratorConfig ;

class RooSegmentedIntegrator2D : public RooSegmentedIntegrator1D {
public:

  // Constructors, assignment etc
  RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, RooIntegrator1D::SummationRule rule=RooIntegrator1D::Trapezoid,
		  Int_t maxSteps= 0, Double_t eps= 0) ; 
  RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, const RooIntegratorConfig& config) ;
  RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax,
		  RooIntegrator1D::SummationRule rule= RooIntegrator1D::Trapezoid, Int_t maxSteps= 0, Double_t eps= 0) ; 
  RooSegmentedIntegrator2D(const RooAbsFunc& function, Int_t nseg, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax,
		  const RooIntegratorConfig& config) ;
  virtual ~RooSegmentedIntegrator2D() ;

protected:

  RooSegmentedIntegrator1D* _xIntegrator ;
  RooAbsFunc* _xint ;

  ClassDef(RooSegmentedIntegrator2D,0) // 1-dimensional numerical integration engine
};

#endif
