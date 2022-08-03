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
  RooAbsIntegrator(const RooAbsFunc& function, bool printEvalCounter=false);
  /// Destructor
  inline ~RooAbsIntegrator() override {
  }
  virtual RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const = 0 ;

  /// Is integrator in valid state
  inline bool isValid() const {
    return _valid;
  }

  /// Return value of integrand at given observable values
  inline double integrand(const double x[]) const {
    return (*_function)(x);
  }

  /// Return integrand function binding
  inline const RooAbsFunc *integrand() const {
    return _function;
  }

  /// If true, finite limits are required on the observable range
  inline virtual bool checkLimits() const {
    return true;
  }

  double calculate(const double *yvec=nullptr) ;
  virtual double integral(const double *yvec=nullptr)=0 ;

  virtual bool canIntegrate1D() const = 0 ;
  virtual bool canIntegrate2D() const = 0 ;
  virtual bool canIntegrateND() const = 0 ;
  virtual bool canIntegrateOpenEnded() const = 0 ;

  bool printEvalCounter() const { return _printEvalCounter ; }
  void setPrintEvalCounter(bool value) { _printEvalCounter = value ; }

  virtual bool setLimits(double*, double*) { return false ; }
  virtual bool setLimits(double xmin, double xmax) ;
  virtual bool setUseIntegrandLimits(bool flag) ;

protected:

  const RooAbsFunc *_function; ///< Pointer to function binding of integrand
  bool _valid;               ///< Is integrator in valid state?
  bool _printEvalCounter ;   ///< If true print number of function evaluation required for integration

  ClassDefOverride(RooAbsIntegrator,0) // Abstract interface for real-valued function integrators
};

#endif
