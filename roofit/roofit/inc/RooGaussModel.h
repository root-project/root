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

  enum RooGaussBasis { noBasis=0, expBasisMinus= 1, expBasisSum= 2, expBasisPlus= 3,
                                  sinBasisMinus=11, sinBasisSum=12, sinBasisPlus=13,
                                  cosBasisMinus=21, cosBasisSum=22, cosBasisPlus=23,
                                                                    linBasisPlus=33,
                                                                   quadBasisPlus=43,
              coshBasisMinus=51,coshBasisSum=52,coshBasisPlus=53,
                 sinhBasisMinus=61,sinhBasisSum=62,sinhBasisPlus=63};
  enum BasisType { none=0, expBasis=1, sinBasis=2, cosBasis=3,
         linBasis=4, quadBasis=5, coshBasis=6, sinhBasis=7 } ;
  enum BasisSign { Both=0, Plus=+1, Minus=-1 } ;

  // Constructors, assignment etc
  inline RooGaussModel() : _flatSFInt(false), _asympInt(false) { }
  RooGaussModel(const char *name, const char *title, RooAbsRealLValue& x,
      RooAbsReal& mean, RooAbsReal& sigma) ;
  RooGaussModel(const char *name, const char *title, RooAbsRealLValue& x,
      RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& msSF) ;
  RooGaussModel(const char *name, const char *title, RooAbsRealLValue& x,
      RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& meanSF, RooAbsReal& sigmaSF) ;
  RooGaussModel(const RooGaussModel& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooGaussModel(*this,newname) ; }
  ~RooGaussModel() override;

  Int_t basisCode(const char* name) const override ;
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  void advertiseFlatScaleFactorIntegral(bool flag) { _flatSFInt = flag ; }

  void advertiseAymptoticIntegral(bool flag) { _asympInt = flag ; }  // added FMV,07/24/03

protected:

  Double_t evaluate() const override ;

  // Calculate common normalization factors
  std::complex<Double_t> evalCerfInt(Double_t sign, Double_t wt, Double_t tau, Double_t umin, Double_t umax, Double_t c) const;

  bool _flatSFInt ;

  bool _asympInt ;  // added FMV,07/24/03

  RooRealProxy mean ;
  RooRealProxy sigma ;
  RooRealProxy msf ;
  RooRealProxy ssf ;

  ClassDefOverride(RooGaussModel,1) // Gaussian Resolution Model
};

#endif
