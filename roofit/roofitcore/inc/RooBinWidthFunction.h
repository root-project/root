// Author Stephan Hageboeck, CERN, 6/2020
/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2020, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOFIT_ROOFITCORE_INC_BINWIDTHFUNCTION_H_
#define ROOFIT_ROOFITCORE_INC_BINWIDTHFUNCTION_H_

#include "RooAbsReal.h"
#include "RooTemplateProxy.h"
#include "RooHistFunc.h"

class RooBinWidthFunction : public RooAbsReal {
  static bool _enabled;

public:
  static void enableClass();
  static void disableClass();
  static bool isClassEnabled();

  /// Create an empty instance.
  RooBinWidthFunction() :
    _histFunc("HistFuncForBinWidth", "Handle to a RooHistFunc, whose bin volumes should be returned.", this,
        /*valueServer=*/true, /*shapeServer=*/true) { }

  RooBinWidthFunction(const char* name, const char* title, const RooHistFunc& histFunc, bool divideByBinWidth);

  /// Copy an existing object.
  RooBinWidthFunction(const RooBinWidthFunction& other, const char* newname = nullptr) :
    RooAbsReal(other, newname),
    _histFunc("HistFuncForBinWidth", this, other._histFunc),
    _divideByBinWidth(other._divideByBinWidth) { }

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

  /// Copy the object and return as TObject*.
  TObject* clone(const char* newname = nullptr) const override {
    return new RooBinWidthFunction(*this, newname);
  }

  // Plotting and binning hints
  /// Test if internal RooHistFunc is binned.
  bool isBinnedDistribution(const RooArgSet& obs) const override {
    return _histFunc->isBinnedDistribution(obs);
  }
  /// Return bin boundaries of internal RooHistFunc.
  std::list<double>* binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const override {
    return _histFunc->binBoundaries(obs, xlo, xhi);
  }
  /// Return plotSamplingHint of internal RooHistFunc.
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override {
    return _histFunc->plotSamplingHint(obs, xlo, xhi);
  }

  bool divideByBinWidth() const { return _divideByBinWidth; }
  const RooHistFunc& histFunc() const { return (*_histFunc); }
  double evaluate() const override;
  void doEval(RooFit::EvalContext &) const override;

private:
  RooTemplateProxy<const RooHistFunc> _histFunc;
  bool _divideByBinWidth{false};

  ClassDefOverride(RooBinWidthFunction, 1);
};

#endif
