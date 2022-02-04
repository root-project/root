/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsIntegrator.h,v 1.18 2007/05/11 09:11:30 verkerke Exp $
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
  /// Destructor
  inline ~RooAbsIntegrator() override {
  }
  virtual RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const = 0 ;

  /// Is integrator in valid state
  inline Bool_t isValid() const {
    return _valid;
  }

  /// Return value of integrand at given observable values
  inline Double_t integrand(const Double_t x[]) const {
    return (*_function)(x);
  }

  /// Return integrand function binding
  inline const RooAbsFunc *integrand() const {
    return _function;
  }

  /// If true, finite limits are required on the observable range
  inline virtual Bool_t checkLimits() const {
    return kTRUE;
  }

  Double_t calculate(const Double_t *yvec=0) ;
  virtual Double_t integral(const Double_t *yvec=0)=0 ;

  virtual Bool_t canIntegrate1D() const = 0 ;
  virtual Bool_t canIntegrate2D() const = 0 ;
  virtual Bool_t canIntegrateND() const = 0 ;
  virtual Bool_t canIntegrateOpenEnded() const = 0 ;

  Bool_t printEvalCounter() const { return _printEvalCounter ; }
  void setPrintEvalCounter(Bool_t value) { _printEvalCounter = value ; }

  virtual Bool_t setLimits(Double_t*, Double_t*) { return kFALSE ; }
  virtual Bool_t setLimits(Double_t xmin, Double_t xmax) ;
  virtual Bool_t setUseIntegrandLimits(Bool_t flag) ;

protected:

  const RooAbsFunc *_function; ///< Pointer to function binding of integrand
  Bool_t _valid;               ///< Is integrator in valid state?
  Bool_t _printEvalCounter ;   ///< If true print number of function evaluation required for integration

  ClassDefOverride(RooAbsIntegrator,0) // Abstract interface for real-valued function integrators
};

#endif
