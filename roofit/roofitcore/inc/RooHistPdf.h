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
#include "RooDataHist.h"

#include <list>

class RooRealVar;
class RooAbsReal;

class RooHistPdf : public RooAbsPdf {
public:
  RooHistPdf() {}
  RooHistPdf(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistPdf(const char *name, const char *title, const RooArgList& pdfObs, const RooArgList& histObs, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistPdf(const char *name, const char *title, const RooArgSet& vars,
             std::unique_ptr<RooDataHist> dhist, int intOrder=0);
  RooHistPdf(const char *name, const char *title, const RooArgList& pdfObs, const RooArgList& histObs,
             std::unique_ptr<RooDataHist> dhist, int intOrder=0);
  RooHistPdf(const RooHistPdf& other, const char* name=nullptr);
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

  static double analyticalIntegral(Int_t code,
                                     const char* rangeName,
                                     RooArgSet const& histObsList,
                                     RooSetProxy const& pdfObsList,
                                     RooDataHist& dataHist,
                                     bool histFuncMode) ;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

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
  double maxVal(Int_t code) const override ;

  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;
  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  bool isBinnedDistribution(const RooArgSet&) const override { return _intOrder==0 ; }


protected:

  bool areIdentical(const RooDataHist& dh1, const RooDataHist& dh2) ;

  bool importWorkspaceHook(RooWorkspace& ws) override ;

  double evaluate() const override;
  double totalVolume() const ;
  friend class RooAbsCachedPdf ;
  double totVolume() const ;

  RooArgSet _histObsList;                      ///< List of observables defining dimensions of histogram
  RooSetProxy _pdfObsList;                     ///< List of observables mapped onto histogram observables
  RooDataHist* _dataHist = nullptr;            ///< Unowned pointer to underlying histogram
  std::unique_ptr<RooDataHist> _ownedDataHist; ///<! Owned pointer to underlying histogram
  mutable RooAICRegistry _codeReg ;            ///<! Auxiliary class keeping tracking of analytical integration code
  Int_t _intOrder = 0;                         ///< Interpolation order
  bool _cdfBoundaries = false;                 ///< Use boundary conditions for CDFs.
  mutable double _totVolume = 0.0;             ///<! Total volume of space (product of ranges of observables)
  bool _unitNorm  = false;                     ///< Assume contents is unit normalized (for use as pdf cache)

  ClassDefOverride(RooHistPdf,4) // Histogram based PDF
};

#endif
