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
#ifndef ROO_BIN_INTEGRATOR
#define ROO_BIN_INTEGRATOR

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"
#include <vector>
#include <list>

namespace RooBatchCompute {
struct RunContext;
}

class RooBinIntegrator : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  RooBinIntegrator() ;

  RooBinIntegrator(const RooAbsFunc& function) ;
  RooBinIntegrator(const RooAbsFunc& function, const RooNumIntConfig& config) ;

  RooAbsIntegrator* clone(const RooAbsFunc& function, const RooNumIntConfig& config) const override ;
  ~RooBinIntegrator() override;

  bool checkLimits() const override;
  Double_t integral(const Double_t *yvec=0) override ;

  using RooAbsIntegrator::setLimits ;
  bool setLimits(Double_t* xmin, Double_t* xmax) override;
  bool setUseIntegrandLimits(bool flag) override {_useIntegrandLimits = flag ; return true ; }

  bool canIntegrate1D() const override { return true ; }
  bool canIntegrate2D() const override { return true ; }
  bool canIntegrateND() const override { return true ; }
  bool canIntegrateOpenEnded() const override { return false ; }

protected:

  friend class RooNumIntFactory ;
  static void registerIntegrator(RooNumIntFactory& fact) ;
  RooBinIntegrator(const RooBinIntegrator&) ;

  // Numerical integrator workspace
  mutable std::vector<Double_t> _xmin;      ///<! Lower integration bound
  mutable std::vector<Double_t> _xmax;      ///<! Upper integration bound
  std::vector<std::vector<double>> _binb;   ///<! list of bin boundaries
  mutable Int_t _numBins;                   ///<! Size of integration range

  bool _useIntegrandLimits;  ///< If true limits of function binding are ued

  std::unique_ptr<RooBatchCompute::RunContext> _evalData;     ///<! Run context for evaluating a function.
  std::unique_ptr<RooBatchCompute::RunContext> _evalDataOrig; ///<! Run context to save bin centres in between invocations.

  double* xvec(double xx) { _x[0] = xx ; return _x ; }
  double* xvec(double xx, double yy) { _x[0] = xx ; _x[1] = yy ; return _x ; }
  double* xvec(double xx, double yy, double zz) { _x[0] = xx ; _x[1] = yy ; _x[2] = zz ; return _x ; }

  Double_t *_x ; ///<! do not persist

  ClassDefOverride(RooBinIntegrator,0) // 1-dimensional numerical integration engine
};

#endif
