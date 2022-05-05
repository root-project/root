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
#ifndef ROO_HIST_PDF
#define ROO_HIST_PDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"
#include <list>

class RooRealVar;
class RooAbsReal;
class RooDataHist ;

class RooHistPdf : public RooAbsPdf {
public:
  RooHistPdf() ;
  RooHistPdf(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistPdf(const char *name, const char *title, const RooArgList& pdfObs, const RooArgList& histObs, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistPdf(const RooHistPdf& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooHistPdf(*this,newname); }
  ~RooHistPdf() override ;

  RooDataHist& dataHist()  {
    // Return RooDataHist that is represented
    return *_dataHist ;
  }
  const RooDataHist& dataHist() const {
    // Return RooDataHist that is represented
    return *_dataHist ;
  }

  void setInterpolationOrder(Int_t order) {
    // Set histogram interpolation order
    _intOrder = order ;
  }
  Int_t getInterpolationOrder() const {
    // Return histogram interpolation order
    return _intOrder ;
  }

  static Int_t getAnalyticalIntegral(RooArgSet& allVars,
                                     RooArgSet& analVars,
                                     const char* rangeName,
                                     RooArgSet const& histObsList,
                                     RooSetProxy const& pdfObsList,
                                     Int_t intOrder) ;

  static Double_t analyticalIntegral(Int_t code,
                                     const char* rangeName,
                                     RooArgSet const& histObsList,
                                     RooSetProxy const& pdfObsList,
                                     RooDataHist& dataHist,
                                     bool histFuncMode) ;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  void setCdfBoundaries(bool flag) {
    // Set use of special boundary conditions for c.d.f.s
    _cdfBoundaries = flag ;
  }
  bool getCdfBoundaries() const {
    // If true, special boundary conditions for c.d.f.s are used
    return _cdfBoundaries ;
  }

  void setUnitNorm(bool flag) {
    // Declare contents to have unit normalization
    _unitNorm = flag ;
  }
  bool haveUnitNorm() const {
    // Return true if contents is declared to be unit normalized
    return _unitNorm ;
  }

  bool selfNormalized() const override { return _unitNorm ; }

  Int_t getMaxVal(const RooArgSet& vars) const override ;
  Double_t maxVal(Int_t code) const override ;

  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const override ;
  std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;
  bool isBinnedDistribution(const RooArgSet&) const override { return _intOrder==0 ; }


protected:

  bool areIdentical(const RooDataHist& dh1, const RooDataHist& dh2) ;

  bool importWorkspaceHook(RooWorkspace& ws) override ;

  Double_t evaluate() const override;
  Double_t totalVolume() const ;
  friend class RooAbsCachedPdf ;
  Double_t totVolume() const ;

  RooArgSet         _histObsList ;   ///< List of observables defining dimensions of histogram
  RooSetProxy       _pdfObsList ;    ///< List of observables mapped onto histogram observables
  RooDataHist*      _dataHist ;      ///< Unowned pointer to underlying histogram
  mutable RooAICRegistry _codeReg ;  ///<! Auxiliary class keeping tracking of analytical integration code
  Int_t             _intOrder ;      ///< Interpolation order
  bool            _cdfBoundaries ; ///< Use boundary conditions for CDFs.
  mutable Double_t  _totVolume ;     ///<! Total volume of space (product of ranges of observables)
  bool            _unitNorm  ;     ///< Assume contents is unit normalized (for use as pdf cache)

  ClassDefOverride(RooHistPdf,4) // Histogram based PDF
};

#endif
