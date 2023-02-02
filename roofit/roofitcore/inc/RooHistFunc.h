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
#include "RooDataHist.h"

#include <list>

class RooRealVar;
class RooAbsReal;

class RooHistFunc : public RooAbsReal {
public:
  RooHistFunc() {}
  RooHistFunc(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistFunc(const char *name, const char *title, const RooArgList& funcObs, const RooArgList& histObs, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistFunc(const char *name, const char *title, const RooArgSet& vars,
             std::unique_ptr<RooDataHist> dhist, int intOrder=0);
  RooHistFunc(const char *name, const char *title, const RooArgList& pdfObs, const RooArgList& histObs,
             std::unique_ptr<RooDataHist> dhist, int intOrder=0);
  RooHistFunc(const RooHistFunc& other, const char* name=nullptr);
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

  /// Replaces underlying RooDataHist with a clone, which is now owned, and returns the clone.
  /// If the underlying RooDataHist is already owned, then that is returned instead of being cloned.
  RooDataHist* cloneAndOwnDataHist(const char* newname="");

  /// Get total bin volume spanned by this hist function.
  /// In 1-d, this is e.g. the range spanned on the x-axis.
  double totVolume() const;

  /// Set histogram interpolation order.
  void setInterpolationOrder(Int_t order) {

    _intOrder = order ;
  }

  /// Return histogram interpolation order.
  Int_t getInterpolationOrder() const {

    return _intOrder ;
  }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  /// Set use of special boundary conditions for c.d.f.s
  void setCdfBoundaries(bool flag) {
    _cdfBoundaries = flag ;
  }

  /// If true, special boundary conditions for c.d.f.s are used
  bool getCdfBoundaries() const {

    return _cdfBoundaries ;
  }

  Int_t getMaxVal(const RooArgSet& vars) const override;
  double maxVal(Int_t code) const override;

  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;
  bool isBinnedDistribution(const RooArgSet&) const override { return _intOrder==0 ; }
  RooArgSet const& getHistObsList() const { return _histObsList; }


  Int_t getBin() const;
  std::vector<Int_t> getBins(RooFit::Detail::DataMap const& dataMap) const;

protected:

  bool importWorkspaceHook(RooWorkspace& ws) override ;
  bool areIdentical(const RooDataHist& dh1, const RooDataHist& dh2) ;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t size, RooFit::Detail::DataMap const&) const override;
  friend class RooAbsCachedReal ;

  void ioStreamerPass2() override ;

  RooArgSet _histObsList;                      ///< List of observables defining dimensions of histogram
  RooSetProxy _depList;                        ///< List of observables mapped onto histogram observables
  RooDataHist* _dataHist = nullptr;            ///< Unowned pointer to underlying histogram
  std::unique_ptr<RooDataHist> _ownedDataHist; ///<! Owned pointer to underlying histogram
  mutable RooAICRegistry _codeReg;             ///<! Auxiliary class keeping tracking of analytical integration code
  Int_t _intOrder = 0;                         ///< Interpolation order
  bool _cdfBoundaries = false;                 ///< Use boundary conditions for CDFs.
  mutable double _totVolume = 0.0;             ///<! Total volume of space (product of ranges of observables)
  bool _unitNorm = false;                      ///<! Assume contents is unit normalized (for use as pdf cache)

  ClassDefOverride(RooHistFunc,2) // Histogram based function
};

#endif
