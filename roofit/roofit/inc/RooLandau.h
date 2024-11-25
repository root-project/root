/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooLandau.h,v 1.5 2007/07/12 20:30:49 wouter Exp $
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
#ifndef ROO_LANDAU
#define ROO_LANDAU

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooLandau : public RooAbsPdf {
public:
  RooLandau() {} ;
  // Original constructor without RooAbsReal::Ref for backwards compatibility.
  inline RooLandau(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _sigma)
      : RooLandau{name, title, RooAbsReal::Ref{_x}, RooAbsReal::Ref{_mean}, RooAbsReal::Ref{_sigma}} {}
  RooLandau(const char *name, const char *title, RooAbsReal::Ref _x, RooAbsReal::Ref _mean, RooAbsReal::Ref _sigma);
  RooLandau(const RooLandau& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooLandau(*this,newname); }

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
  double analyticalIntegral(Int_t code, const char *rangeName) const override;

  RooAbsReal const& getX() const { return *x; }
  RooAbsReal const& getMean() const { return *mean; }
  RooAbsReal const& getSigma() const { return *sigma; }

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy sigma ;

  double evaluate() const override ;
  void doEval(RooFit::EvalContext &) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:

  ClassDefOverride(RooLandau,1) // Landau Distribution PDF
};

#endif
