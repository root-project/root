/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGaussKronrodIntegrator1D.h,v 1.5 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_GAUSS_KRONROD_INTEGRATOR_1D
#define ROO_GAUSS_KRONROD_INTEGRATOR_1D

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"

double RooGaussKronrodIntegrator1D_GSL_GlueFunction(double x, void *data) ;

class RooGaussKronrodIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooGaussKronrodIntegrator1D() ;
  RooGaussKronrodIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) ;
  RooGaussKronrodIntegrator1D(const RooAbsFunc& function, double xmin, double xmax, const RooNumIntConfig& config) ;
  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooGaussKronrodIntegrator1D() override;

  bool checkLimits() const override;
  double integral(const double *yvec=0) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(double* xmin, double* xmax) override;
  bool setUseIntegrandLimits(bool flag) override {_useIntegrandLimits = flag ; return true ; }

  bool canIntegrate1D() const override { return true ; }
  bool canIntegrate2D() const override { return false ; }
  bool canIntegrateND() const override { return false ; }
  bool canIntegrateOpenEnded() const override { return true ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  friend double RooGaussKronrodIntegrator1D_GSL_GlueFunction(double x, void *data) ;

  bool initialize();

  bool _useIntegrandLimits;  // Use limits in function binding?

  double* xvec(double& xx) { _x[0] = xx ; return _x ; }
  double *_x ; //! do not persist

  double _epsAbs ;                   // Absolute precision
  double _epsRel ;                   // Relative precision

  mutable double _xmin;              //! Lower integration bound
  mutable double _xmax;              //! Upper integration bound

  ClassDefOverride(RooGaussKronrodIntegrator1D,0) // 1-dimensional Gauss-Kronrod numerical integration engine
};

#endif
