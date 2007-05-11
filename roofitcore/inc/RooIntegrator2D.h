/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooIntegrator2D.rdl,v 1.7 2006/07/03 15:37:11 wverkerke Exp $
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
#ifndef ROO_INTEGRATOR_2D
#define ROO_INTEGRATOR_2D

#include "RooIntegrator1D.h"
#include "RooNumIntConfig.h"

class RooIntegrator2D : public RooIntegrator1D {
public:

  // Constructors, assignment etc
  RooIntegrator2D() ;
  RooIntegrator2D(const RooAbsFunc& function, RooIntegrator1D::SummationRule rule=RooIntegrator1D::Trapezoid,
		  Int_t maxSteps= 0, Double_t eps= 0) ; 
  RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax,
		  SummationRule rule= Trapezoid, Int_t maxSteps= 0, Double_t eps= 0) ; 

  RooIntegrator2D(const RooAbsFunc& function, const RooNumIntConfig& config) ;
  RooIntegrator2D(const RooAbsFunc& function, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax,
		  const RooNumIntConfig& config) ;

  virtual RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const ;
  virtual ~RooIntegrator2D() ;

  virtual Bool_t checkLimits() const;

  virtual Bool_t canIntegrate1D() const { return kFALSE ; }
  virtual Bool_t canIntegrate2D() const { return kTRUE ; }
  virtual Bool_t canIntegrateND() const { return kFALSE ; }
  virtual Bool_t canIntegrateOpenEnded() const { return kFALSE ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;	

  RooIntegrator1D* _xIntegrator ;
  RooAbsFunc* _xint ;

  ClassDef(RooIntegrator2D,0) // 1-dimensional numerical integration engine
};

#endif
