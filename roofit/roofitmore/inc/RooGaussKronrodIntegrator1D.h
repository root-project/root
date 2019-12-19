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
  RooGaussKronrodIntegrator1D(const RooAbsFunc& function, Double_t xmin, Double_t xmax, const RooNumIntConfig& config) ;
  virtual RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const ;
  virtual ~RooGaussKronrodIntegrator1D();

  virtual Bool_t checkLimits() const;
  virtual Double_t integral(const Double_t *yvec=0) ;

  using RooAbsIntegrator::setLimits ;
  Bool_t setLimits(Double_t* xmin, Double_t* xmax);
  virtual Bool_t setUseIntegrandLimits(Bool_t flag) {_useIntegrandLimits = flag ; return kTRUE ; }

  virtual Bool_t canIntegrate1D() const { return kTRUE ; }
  virtual Bool_t canIntegrate2D() const { return kFALSE ; }
  virtual Bool_t canIntegrateND() const { return kFALSE ; }
  virtual Bool_t canIntegrateOpenEnded() const { return kTRUE ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;	

  friend double RooGaussKronrodIntegrator1D_GSL_GlueFunction(double x, void *data) ;

  Bool_t initialize();

  Bool_t _useIntegrandLimits;  // Use limits in function binding?

  Double_t* xvec(Double_t& xx) { _x[0] = xx ; return _x ; }
  Double_t *_x ; //! do not persist

  Double_t _epsAbs ;                   // Absolute precision
  Double_t _epsRel ;                   // Relative precision

  mutable Double_t _xmin;              //! Lower integration bound
  mutable Double_t _xmax;              //! Upper integration bound

  ClassDef(RooGaussKronrodIntegrator1D,0) // 1-dimensional Gauss-Kronrod numerical integration engine
};

#endif
