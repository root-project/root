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
#include "RooListProxy.h"

#include <utility>
#include <vector>

class RooHistFunc;
class RooAbsBinning;

class RooBinWidthFunction : public RooAbsReal {
  static bool _enabled;

public:
  static void enableClass();
  static void disableClass();
  static bool isClassEnabled();

  /// Create an empty instance.
  RooBinWidthFunction() = default;

  RooBinWidthFunction(const char* name, const char* title, const RooArgList& observables, bool divideByBinWidth);

  RooBinWidthFunction(const char* name, const char* title, const RooHistFunc& histFunc, bool divideByBinWidth);

  /// Copy an existing object.
  RooBinWidthFunction(const RooBinWidthFunction& other, const char* newname = nullptr) :
    RooAbsReal(other, newname),
    _observables(this, other._observables),
    _divideByBinWidth(other._divideByBinWidth) { }

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

  /// Copy the object and return as TObject*.
  TObject* clone(const char* newname = nullptr) const override {
    return new RooBinWidthFunction(*this, newname);
  }

  // Plotting and binning hints
  bool isBinnedDistribution(const RooArgSet&) const override { return true; }
  std::list<double>* binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const override;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override;

  bool divideByBinWidth() const { return _divideByBinWidth; }
  const RooArgList& variables() const { return _observables; }
  double getValV(const RooArgSet* nset = nullptr) const override;
  double evaluate() const override;
  void doEval(RooFit::EvalContext &) const override;

protected:
  void ioStreamerPass2() override;

private:
  void finalizeIO();
  bool updateCache() const;

  RooListProxy _observables{"observables", "Observables defining the bin volume", this, true, true};
  mutable std::vector<double> _binVolumes; //! Cached bin volumes, in observable order
  mutable std::vector<int> _binCounts; //! Number of bins per observable
  mutable std::vector<const RooAbsBinning*> _binnings; //! Binnings used to build the cache
  mutable std::vector<std::pair<double, double>> _binRanges; //! Ranges used to build the cache
  bool _divideByBinWidth{false};

  ClassDefOverride(RooBinWidthFunction, 2);
};

#endif
