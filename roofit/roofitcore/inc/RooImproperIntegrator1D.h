/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooImproperIntegrator1D.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_IMPROPER_INTEGRATOR_1D
#define ROO_IMPROPER_INTEGRATOR_1D

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"

class RooInvTransform;
class RooIntegrator1D;

class RooImproperIntegrator1D : public RooAbsIntegrator {
public:

  RooImproperIntegrator1D() ;
  RooImproperIntegrator1D(const RooAbsFunc& function);
  RooImproperIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config);
  RooImproperIntegrator1D(const RooAbsFunc& function, double xmin, double xmax, const RooNumIntConfig& config);
  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;

  bool checkLimits() const override;
  using RooAbsIntegrator::setLimits ;
  bool setLimits(double* xmin, double* xmax) override;
  bool setUseIntegrandLimits(bool flag) override {_useIntegrandLimits = flag ; return true ; }
  double integral(const double* yvec=0) override ;

  bool canIntegrate1D() const override { return true ; }
  bool canIntegrate2D() const override { return false ; }
  bool canIntegrateND() const override { return false ; }
  bool canIntegrateOpenEnded() const override { return true ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  void initialize(const RooAbsFunc* function=0) ;

  enum LimitsCase { Invalid, ClosedBothEnds, OpenBothEnds, OpenBelowSpansZero, OpenBelow,
          OpenAboveSpansZero, OpenAbove };
  LimitsCase limitsCase() const;
  LimitsCase _case; ///< Configuration of limits
  mutable double _xmin, _xmax; ///< Value of limits
  bool _useIntegrandLimits;    ///< Use limits in function binding?

  RooAbsFunc*      _origFunc = nullptr;  ///< Original function binding
  std::unique_ptr<RooInvTransform> _function;   ///< Binding with inverse of function
  RooNumIntConfig  _config ;    ///< Configuration object
  mutable std::unique_ptr<RooIntegrator1D> _integrator1; ///< Piece integrator 1
  mutable std::unique_ptr<RooIntegrator1D> _integrator2; ///< Piece integrator 2
  mutable std::unique_ptr<RooIntegrator1D> _integrator3; ///< Piece integrator 3

  ClassDefOverride(RooImproperIntegrator1D,0) // 1-dimensional improper integration engine
};

#endif
