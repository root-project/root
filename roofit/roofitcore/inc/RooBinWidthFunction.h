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

namespace BatchHelpers { struct RunContext; }

class RooBinWidthFunction : public RooAbsReal {
public:
  /// Create an empty instance.
  RooBinWidthFunction() :
    _histFunc("HistFuncForBinWidth", "Handle to a RooHistFunc, whose bin volumes should be returned.", this,
        /*valueServer=*/true, /*shapeServer=*/true) { }

  /// Create an instance.
  /// \param name Name to identify the object.
  /// \param title Title for e.g. plotting.
  /// \param histFunc RooHistFunc object whose bin widths should be returned.
  /// \param divideByBinWidth If true, return inverse bin width.
  RooBinWidthFunction(const char* name, const char* title, const RooHistFunc& histFunc, bool divideByBinWidth) :
    RooAbsReal(name, title),
    _histFunc("HistFuncForBinWidth", "Handle to a RooHistFunc, whose bin volumes should be returned.", this, histFunc, /*valueServer=*/true, /*shapeServer=*/true),
    _divideByBinWidth(divideByBinWidth) { }

  /// Copy an existing object.
  RooBinWidthFunction(const RooBinWidthFunction& other, const char* newname = nullptr) :
    RooAbsReal(other, newname),
    _histFunc("HistFuncForBinWidth", this, other._histFunc),
    _divideByBinWidth(other._divideByBinWidth) { }

  ~RooBinWidthFunction() override { }

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
  std::list<Double_t>* binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const override {
    return _histFunc->binBoundaries(obs, xlo, xhi);
  }
  /// Return plotSamplingHint of internal RooHistFunc.
  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const override {
    return _histFunc->plotSamplingHint(obs, xlo, xhi);
  }

  bool divideByBinWidth() const { return _divideByBinWidth; }
  const RooHistFunc& histFunc() const { return (*_histFunc); }
  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t size, RooBatchCompute::DataMap&) const override;

private:
  RooTemplateProxy<const RooHistFunc> _histFunc;
  bool _divideByBinWidth{false};

  ClassDefOverride(RooBinWidthFunction, 1);
};

#endif
