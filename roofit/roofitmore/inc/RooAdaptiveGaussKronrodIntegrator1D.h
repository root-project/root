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
  RooAdaptiveGaussKronrodIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
                                      const RooNumIntConfig& config) ;
  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooAdaptiveGaussKronrodIntegrator1D() override;

  bool checkLimits() const override;
  double integral(const double *yvec=nullptr) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(double* xmin, double* xmax) override;
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

  double* xvec(double& xx) {
    // Return contents of xx in internal array pointer
    _x[0] = xx ; return _x ;
  }
  double *_x ;                        //! Current coordinate

  double _epsAbs ;                   // Absolute precision
  double _epsRel ;                   // Relative precision
  Int_t    _methodKey ;                // GSL method key
  Int_t    _maxSeg ;                   // Maximum number of segments
  void*    _workspace ;                // GSL workspace

  mutable double _xmin;              //! Lower integration bound
  mutable double _xmax;              //! Upper integration bound

  ClassDefOverride(RooAdaptiveGaussKronrodIntegrator1D,0) // 1-dimensional adaptive Gauss-Kronrod numerical integration engine
};

#endif
