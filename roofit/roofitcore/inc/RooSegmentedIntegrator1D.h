/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSegmentedIntegrator1D.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SEGMENTED_INTEGRATOR_1D
#define ROO_SEGMENTED_INTEGRATOR_1D

#include "RooAbsIntegrator.h"
#include "RooIntegrator1D.h"
#include "RooNumIntConfig.h"

class RooSegmentedIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooSegmentedIntegrator1D() ;
  RooSegmentedIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) ;
  RooSegmentedIntegrator1D(const RooAbsFunc& function, double xmin, double xmax, const RooNumIntConfig& config) ;

  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooSegmentedIntegrator1D() override;

  bool checkLimits() const override;
  double integral(const double *yvec=nullptr) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(double *xmin, double *xmax) override;
  bool setUseIntegrandLimits(bool flag) override { _useIntegrandLimits = flag ; return true ; }

  bool canIntegrate1D() const override { return true ; }
  bool canIntegrate2D() const override { return false ; }
  bool canIntegrateND() const override { return false ; }
  bool canIntegrateOpenEnded() const override { return false ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  mutable double _xmin ;
  mutable double _xmax ;
  mutable double _range ;
  bool _valid ;
  Int_t _nseg ; // Number of segments
  bool _useIntegrandLimits ;

  RooNumIntConfig _config ;
  RooIntegrator1D** _array ; ///< Array of segment integrators

  bool initialize();

  ClassDefOverride(RooSegmentedIntegrator1D,0) // 1-dimensional piece-wise numerical integration engine
};

#endif
