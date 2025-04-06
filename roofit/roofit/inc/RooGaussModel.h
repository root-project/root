/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooGaussModel.h,v 1.21 2007/05/11 09:13:07 verkerke Exp $
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
#ifndef ROO_GAUSS_MODEL
#define ROO_GAUSS_MODEL

#include "RooResolutionModel.h"
#include "RooRealProxy.h"

#include <cmath>
#include <complex>

class RooGaussModel : public RooResolutionModel {
public:
  // Constructors, assignment etc
  RooGaussModel() = default;
  RooGaussModel(const char *name, const char *title, RooAbsRealLValue& x,
      RooAbsReal& mean, RooAbsReal& sigma) ;
  RooGaussModel(const char *name, const char *title, RooAbsRealLValue& x,
      RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& msSF) ;
  RooGaussModel(const char *name, const char *title, RooAbsRealLValue& x,
      RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& meanSF, RooAbsReal& sigmaSF) ;
  RooGaussModel(const RooGaussModel& other, const char* name=nullptr);
  TObject* clone(const char* newname=nullptr) const override { return new RooGaussModel(*this,newname) ; }

  Int_t basisCode(const char* name) const override ;
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  void advertiseFlatScaleFactorIntegral(bool flag) { _flatSFInt = flag ; }

  void advertiseAymptoticIntegral(bool flag) { _asympInt = flag ; }  // added FMV,07/24/03

  void doEval(RooFit::EvalContext &) const override;

  bool canComputeBatchWithCuda() const override;

protected:

  double evaluate() const override ;
  static double evaluate(double x, double mean, double sigma, double param1, double param2, int basisCode);

  // Calculate common normalization factors
  std::complex<double> evalCerfInt(double sign, double wt, double tau, double umin, double umax, double c) const;

private:
  bool _flatSFInt = false;

  bool _asympInt = false;  // added FMV,07/24/03

  RooRealProxy mean ;
  RooRealProxy sigma ;
  RooRealProxy msf ;
  RooRealProxy ssf ;

  ClassDefOverride(RooGaussModel,1) // Gaussian Resolution Model
};

#endif
