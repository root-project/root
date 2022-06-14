/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_ADAPTIVE_INTEGRATOR_ND
#define ROO_ADAPTIVE_INTEGRATOR_ND

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"
#include "TString.h"

namespace ROOT { namespace Math { class AdaptiveIntegratorMultiDim ; } }
class RooMultiGenFunction ;

class RooAdaptiveIntegratorND : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooAdaptiveIntegratorND() ;
  RooAdaptiveIntegratorND(const RooAbsFunc& function, const RooNumIntConfig& config) ;

  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooAdaptiveIntegratorND() override;

  bool checkLimits() const override;
  double integral(const double *yvec=0) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(double* xmin, double* xmax) override;

  bool canIntegrate1D() const override { return false ; }
  bool canIntegrate2D() const override { return true ; }
  bool canIntegrateND() const override { return true ; }
  bool canIntegrateOpenEnded() const override { return false ; }

  bool setUseIntegrandLimits(bool flag) override {_useIntegrandLimits = flag ; return true ; }

protected:

  RooAdaptiveIntegratorND(const RooAdaptiveIntegratorND&) ;

  bool _useIntegrandLimits;  ///< If true limits of function binding are used

  mutable double* _xmin ;    ///< Lower bound in each dimension
  mutable double* _xmax ;    ///< Upper bound in each dimension
  double _epsRel ;           ///< Relative precision
  double _epsAbs ;           ///< Absolute precision
  Int_t    _nmax ;             ///< Max number of divisions
  Int_t    _nError ;           ///< Number of error occurrences
  Int_t    _nWarn ;            ///< Max number of warnings to be issued ;
  RooMultiGenFunction* _func ; ///<! ROOT::Math multi-parameter function binding
  ROOT::Math::AdaptiveIntegratorMultiDim* _integrator ;
  TString _intName ;           ///< Integrand name

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  ClassDefOverride(RooAdaptiveIntegratorND,0) // N-dimensional adaptive integration (interface to MathCore integrator)
};

#endif
