/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsIntegrator.rdl,v 1.17 2005/04/18 21:44:21 wverkerke Exp $
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
#ifndef ROO_ABS_INTEGRATOR
#define ROO_ABS_INTEGRATOR

#include "RooAbsFunc.h"
#include "RooNumIntConfig.h"

class RooAbsIntegrator : public TObject {
public:
  RooAbsIntegrator() ;
  RooAbsIntegrator(const RooAbsFunc& function, Bool_t printEvalCounter=kFALSE);
  inline virtual ~RooAbsIntegrator() { }
  virtual RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const = 0 ;
  
  inline Bool_t isValid() const { return _valid; }

  inline Double_t integrand(const Double_t x[]) const { return (*_function)(x); }
  inline const RooAbsFunc *integrand() const { return _function; }

  inline virtual Bool_t checkLimits() const { return kTRUE; }

  Double_t calculate(const Double_t *yvec=0) ;
  virtual Double_t integral(const Double_t *yvec=0)=0 ;

  virtual Bool_t canIntegrate1D() const = 0 ;
  virtual Bool_t canIntegrate2D() const = 0 ;
  virtual Bool_t canIntegrateND() const = 0 ;
  virtual Bool_t canIntegrateOpenEnded() const = 0 ;

  Bool_t printEvalCounter() const { return _printEvalCounter ; }
  void setPrintEvalCounter(Bool_t value) { _printEvalCounter = value ; }

  virtual Bool_t setLimits(Double_t xmin, Double_t xmax) ;
  virtual Bool_t setUseIntegrandLimits(Bool_t flag) ;

protected:

  const RooAbsFunc *_function;
  Bool_t _valid;
  Bool_t _printEvalCounter ;

  ClassDef(RooAbsIntegrator,0) // Abstract interface for real-valued function integrators
};

#endif
