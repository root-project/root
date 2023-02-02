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

class RooIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  enum SummationRule { Trapezoid, Midpoint };
  RooIntegrator1D() {}

  RooIntegrator1D(const RooAbsFunc& function, SummationRule rule= Trapezoid,
        Int_t maxSteps= 0, double eps= 0) ;
  RooIntegrator1D(const RooAbsFunc& function, double xmin, double xmax,
        SummationRule rule= Trapezoid, Int_t maxSteps= 0, double eps= 0) ;

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
  Int_t _maxSteps ;      ///< Maximum number of steps
  Int_t _minStepsZero ;  ///< Minimum number of steps to declare convergence to zero
  Int_t _fixSteps ;      ///< Fixed number of steps
  double _epsAbs ;     ///< Absolute convergence tolerance
  double _epsRel ;     ///< Relative convergence tolerance
  bool _doExtrap ;     ///< Apply conversion step?
  enum { _nPoints = 5 };

  // Numerical integrator support functions
  double addTrapezoids(Int_t n) ;
  double addMidpoints(Int_t n) ;
  void extrapolate(Int_t n) ;

  // Numerical integrator workspace
  double _xmin;              ///<! Lower integration bound
  double _xmax;              ///<! Upper integration bound
  double _range;             ///<! Size of integration range
  double _extrapValue;       ///<! Extrapolated value
  double _extrapError;       ///<! Error on extrapolated value
  std::vector<double> _h ;   ///<! Integrator workspace
  std::vector<double> _s ;   ///<! Integrator workspace
  std::vector<double> _c ;   ///<! Integrator workspace
  std::vector<double> _d ;   ///<! Integrator workspace
  double _savedResult;       ///<! Integrator workspace

  double* xvec(double& xx) { _x[0] = xx ; return _x.data(); }

  std::vector<double> _x ; //! do not persist

  ClassDefOverride(RooIntegrator1D,0) // 1-dimensional numerical integration engine
};

#endif
