/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAdaptiveGaussKronrodIntegrator1D.h,v 1.5 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ADAPTIVE_GAUSS_KRONROD_INTEGRATOR_1D
#define ROO_ADAPTIVE_GAUSS_KRONROD_INTEGRATOR_1D

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"

double RooAdaptiveGaussKronrodIntegrator1D_GSL_GlueFunction(double x, void *data) ;

class RooAdaptiveGaussKronrodIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooAdaptiveGaussKronrodIntegrator1D() ;
  RooAdaptiveGaussKronrodIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) ;
  RooAdaptiveGaussKronrodIntegrator1D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
                                      const RooNumIntConfig& config) ;
  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooAdaptiveGaussKronrodIntegrator1D() override;

  bool checkLimits() const override;
  Double_t integral(const Double_t *yvec=0) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(Double_t* xmin, Double_t* xmax) override;
  bool setUseIntegrandLimits(bool flag) override {
    // If flag is true, intergration limits are taken from definition in input function binding
    _useIntegrandLimits = flag ; return true ;
  }

  bool canIntegrate1D() const override {
    // We can integrate 1-dimensional functions
    return true ;
  }
  bool canIntegrate2D() const override {
    // We can not integrate 2-dimensional functions
    return false ;
  }
  bool canIntegrateND() const override {
    // We can not integrate >2-dimensional functions
    return false ;
  }
  bool canIntegrateOpenEnded() const override {
    // We can integrate over open-ended domains
    return true ;
  }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  enum DomainType { Closed, OpenLo, OpenHi, Open } ;
  mutable DomainType _domainType ;

  friend double RooAdaptiveGaussKronrodIntegrator1D_GSL_GlueFunction(double x, void *data) ;

  bool initialize();

  bool _useIntegrandLimits;

  Double_t* xvec(Double_t& xx) {
    // Return contents of xx in internal array pointer
    _x[0] = xx ; return _x ;
  }
  Double_t *_x ;                        //! Current coordinate

  Double_t _epsAbs ;                   // Absolute precision
  Double_t _epsRel ;                   // Relative precision
  Int_t    _methodKey ;                // GSL method key
  Int_t    _maxSeg ;                   // Maximum number of segments
  void*    _workspace ;                // GSL workspace

  mutable Double_t _xmin;              //! Lower integration bound
  mutable Double_t _xmax;              //! Upper integration bound

  ClassDefOverride(RooAdaptiveGaussKronrodIntegrator1D,0) // 1-dimensional adaptive Gauss-Kronrod numerical integration engine
};

#endif
