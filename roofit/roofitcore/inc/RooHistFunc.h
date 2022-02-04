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
#ifndef ROO_HIST_FUNC
#define ROO_HIST_FUNC

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"
#include "RooTrace.h"
#include <list>

class RooRealVar;
class RooAbsReal;
class RooDataHist ;

class RooHistFunc : public RooAbsReal {
public:
  RooHistFunc() ;
  RooHistFunc(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistFunc(const char *name, const char *title, const RooArgList& funcObs, const RooArgList& histObs, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistFunc(const RooHistFunc& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooHistFunc(*this,newname); }
  ~RooHistFunc() override ;

  /// Return RooDataHist that is represented.
  RooDataHist& dataHist()  {
    return *_dataHist ;
  }

  /// Return RooDataHist that is represented.
  const RooDataHist& dataHist() const {
    return *_dataHist ;
  }

  /// Get total bin volume spanned by this hist function.
  /// In 1-d, this is e.g. the range spanned on the x-axis.
  Double_t totVolume() const;

  /// Set histogram interpolation order.
  void setInterpolationOrder(Int_t order) {

    _intOrder = order ;
  }

  /// Return histogram interpolation order.
  Int_t getInterpolationOrder() const {

    return _intOrder ;
  }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  /// Set use of special boundary conditions for c.d.f.s
  void setCdfBoundaries(Bool_t flag) {
    _cdfBoundaries = flag ;
  }

  /// If true, special boundary conditions for c.d.f.s are used
  Bool_t getCdfBoundaries() const {

    return _cdfBoundaries ;
  }

  Int_t getMaxVal(const RooArgSet& vars) const override;
  Double_t maxVal(Int_t code) const override;

  std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;
  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const override ;
  Bool_t isBinnedDistribution(const RooArgSet&) const override { return _intOrder==0 ; }
  RooArgSet const& getHistObsList() const { return _histObsList; }


  Int_t getBin() const;
  std::vector<Int_t> getBins(RooBatchCompute::RunContext& evalData) const;

protected:

  Bool_t importWorkspaceHook(RooWorkspace& ws) override ;
  Bool_t areIdentical(const RooDataHist& dh1, const RooDataHist& dh2) ;

  Double_t evaluate() const override;
  RooSpan<double> evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* /*normSet*/) const override;
  friend class RooAbsCachedReal ;

  void ioStreamerPass2() override ;

  RooArgSet         _histObsList ;   ///< List of observables defining dimensions of histogram
  RooSetProxy       _depList ;       ///< List of observables mapped onto histogram observables
  RooDataHist*      _dataHist ;      ///< Unowned pointer to underlying histogram
  mutable RooAICRegistry _codeReg ;  ///<! Auxiliary class keeping tracking of analytical integration code
  Int_t             _intOrder ;      ///< Interpolation order
  Bool_t            _cdfBoundaries ; ///< Use boundary conditions for CDFs.
  mutable Double_t  _totVolume ;     ///<! Total volume of space (product of ranges of observables)
  Bool_t            _unitNorm  ;     ///<! Assume contents is unit normalized (for use as pdf cache)

  ClassDefOverride(RooHistFunc,2) // Histogram based function
};

#endif
