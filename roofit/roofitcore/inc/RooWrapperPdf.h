// Author: Stephan Hageboeck, CERN
/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2018, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_WRAPPER_PDF
#define ROO_WRAPPER_PDF

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooAbsPdf.h"
#include <list>

class RooWrapperPdf final : public RooAbsPdf {
public:

  RooWrapperPdf() { };
  /// Construct a new RooWrapperPdf.
  /// \param[in] name A name to identify this object.
  /// \param[in] title Title (for e.g. plotting)
  /// \param[in] inputFunction Any RooAbsReal that should be converted into a PDF. Although it's possible
  /// \param[in] selfNormalized The return value the RooAbsPdf::selfNormalized() function for the wrapped PDF object.
  ///            If it is `true`, then no automatic normalization will be
  ///            performed when evaluating the function. In this case, the
  ///            effect RooWrapperPdf is not to change the evaluated values,
  ///            but only to wrap the function in something that is of type
  ///            RooAbsPdf, which can be useful if some interface requires it.
  /// to pass a PDF, it only makes sense for non-PDF functions.
  RooWrapperPdf(const char *name, const char *title, RooAbsReal& inputFunction, bool selfNormalized=false) :
    RooAbsPdf(name, title),
    _func("inputFunction", "Function to be converted into a PDF", this, inputFunction),
    _selfNormalized{selfNormalized} { }

  RooWrapperPdf(const RooWrapperPdf& other, const char *name = nullptr) :
    RooAbsPdf(other, name),
    _func("inputFunction", this, other._func),
    _selfNormalized{other._selfNormalized} { }

  TObject* clone(const char* newname) const override {
    return new RooWrapperPdf(*this, newname);
  }

  bool selfNormalized() const override { return _selfNormalized; }

  // Analytical Integration handling
  bool forceAnalyticalInt(const RooAbsArg& /*dep*/) const override {
     // Just like with other wrapper classes like RooExtendPdf, we can safely
     // use the analytical integration capabilities of the wrapped object,
     // because we don't do any no-linear transformation.
     return true;
  }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,
      const char* rangeName=nullptr) const override {
    return _func->getAnalyticalIntegralWN(allVars, analVars, normSet, rangeName);
  }
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars,
      const char* rangeName=nullptr) const override {
    return _func->getAnalyticalIntegral(allVars, numVars, rangeName);
  }
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const override {
    return _func->analyticalIntegralWN(code, normSet, rangeName);
  }
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override {
    return _func->analyticalIntegral(code, rangeName);
  }


  // Internal toy generation. Since our _func is not a PDF (if it is, it doesn't make sense to use this wrapper),
  // we cannot do anything.
  /// Get specialised generator. Since the underlying function is not a PDF, this will always return zero.
//  Int_t getGenerator(const RooArgSet& /*directVars*/, RooArgSet& /*generateVars*/,
//      bool /*staticInitOK = true*/) const override { return 0; }
//  void initGenerator(Int_t /*code*/) override { }
//  void generateEvent(Int_t /*code*/) override { }
//  bool isDirectGenSafe(const RooAbsArg& /*arg*/) const override { return false; }


  // Hints for optimized brute-force sampling
  Int_t getMaxVal(const RooArgSet& vars) const override {
    return _func.arg().getMaxVal(vars);
  }
  double maxVal(Int_t code) const override {
    return _func.arg().maxVal(code);
  }
  Int_t minTrialSamples(const RooArgSet& arGenObs) const override {
    return _func.arg().minTrialSamples(arGenObs);
  }

  // Plotting and binning hints
  bool isBinnedDistribution(const RooArgSet& obs) const override {
    return _func.arg().isBinnedDistribution(obs);
  }
  std::list<double>* binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const override {
    return _func.arg().binBoundaries(obs, xlo, xhi);
  }
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override {
    return _func.arg().plotSamplingHint(obs, xlo, xhi);
  }



private:
  RooRealProxy _func;
  bool _selfNormalized = false;

  double evaluate() const override {
    return _func;
  }

  ClassDefOverride(RooWrapperPdf,2);
};

#endif
