/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooIntegrator1D.h,v 1.21 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_INTEGRATOR_1D
#define ROO_INTEGRATOR_1D

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"

#include <ROOT/RSpan.hxx>

#include <functional>

namespace RooFit {
namespace Detail {

std::pair<double, int> integrate1d(std::function<double(double)> func, bool doTrapezoid, int maxSteps, int minStepsZero,
                                   int fixSteps, double epsAbs, double epsRel, bool doExtrap, double xmin, double xmax,
                                   std::span<double> hArr, std::span<double> sArr);

} // namespace Detail
} // namespace RooFit

class RooIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  enum SummationRule { Trapezoid, Midpoint };
  RooIntegrator1D() {}

  RooIntegrator1D(const RooAbsFunc& function, SummationRule rule= Trapezoid,
        int maxSteps= 0, double eps= 0) ;
  RooIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
        SummationRule rule= Trapezoid, int maxSteps= 0, double eps= 0) ;

  RooIntegrator1D(const RooAbsFunc& function, const RooNumIntConfig& config) ;
  RooIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
        const RooNumIntConfig& config) ;

  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;

  bool checkLimits() const override;
  double integral(const double *yvec=nullptr) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(double* xmin, double* xmax) override;
  bool setUseIntegrandLimits(bool flag) override {_useIntegrandLimits = flag ; return true ; }

  bool canIntegrate1D() const override { return true ; }
  bool canIntegrate2D() const override { return false ; }
  bool canIntegrateND() const override { return false ; }
  bool canIntegrateOpenEnded() const override { return false ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;

  bool initialize();

  bool _useIntegrandLimits;  ///< If true limits of function binding are used

  // Integrator configuration
  SummationRule _rule;
  int _maxSteps ;          ///< Maximum number of steps
  int _minStepsZero = 999; ///< Minimum number of steps to declare convergence to zero
  int _fixSteps = 0;       ///< Fixed number of steps
  double _epsAbs ;         ///< Absolute convergence tolerance
  double _epsRel ;         ///< Relative convergence tolerance
  bool _doExtrap = true;   ///< Apply conversion step?
  double _xmin;            ///<! Lower integration bound
  double _xmax;            ///<! Upper integration bound

  // Numerical integrator workspace
  std::vector<double> _wksp ;   ///<! Integrator workspace

  double* xvec(double& xx) { _x[0] = xx ; return _x.data(); }

  std::vector<double> _x ; //! do not persist

  ClassDefOverride(RooIntegrator1D,0) // 1-dimensional numerical integration engine
};

#endif
