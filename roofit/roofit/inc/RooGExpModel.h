/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooGExpModel.h,v 1.16 2007/05/11 09:13:07 verkerke Exp $
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
#ifndef ROO_GEXP_MODEL
#define ROO_GEXP_MODEL

#include "Rtypes.h"
#include "RooResolutionModel.h"
#include "RooRealProxy.h"

#include <complex>

class RooGExpModel : public RooResolutionModel {
public:

  enum RooGExpBasis { noBasis=0, expBasisMinus= 1, expBasisSum= 2, expBasisPlus= 3,
                       sinBasisMinus=11, sinBasisSum=12, sinBasisPlus=13,
                                 cosBasisMinus=21, cosBasisSum=22, cosBasisPlus=23,
             sinhBasisMinus=31,sinhBasisSum=32,sinhBasisPlus=33,
             coshBasisMinus=41,coshBasisSum=42,coshBasisPlus=43} ;



  enum BasisType { none=0, expBasis=1, sinBasis=2, cosBasis=3, sinhBasis=4, coshBasis=5 } ;
  enum BasisSign { Both=0, Plus=+1, Minus=-1 } ;
  enum Type { Normal, Flipped };

  // Constructors, assignment etc
  inline RooGExpModel() {
    // coverity[UNINIT_CTOR]
  }

  RooGExpModel(const char *name, const char *title, RooAbsRealLValue& x,
          RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& rlife,
          RooAbsReal& meanSF, RooAbsReal& sigmaSF, RooAbsReal& rlifeSF,
          Bool_t nlo=kFALSE, Type type=Normal) ;

  RooGExpModel(const char *name, const char *title, RooAbsRealLValue& x,
          RooAbsReal& sigma, RooAbsReal& rlife,
          Bool_t nlo=kFALSE, Type type=Normal) ;

  RooGExpModel(const char *name, const char *title, RooAbsRealLValue& x,
          RooAbsReal& sigma, RooAbsReal& rlife,
          RooAbsReal& srSF,
          Bool_t nlo=kFALSE, Type type=Normal) ;

  RooGExpModel(const char *name, const char *title, RooAbsRealLValue& x,
          RooAbsReal& sigma, RooAbsReal& rlife,
          RooAbsReal& sigmaSF, RooAbsReal& rlifeSF,
          Bool_t nlo=kFALSE, Type type=Normal) ;



  RooGExpModel(const RooGExpModel& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooGExpModel(*this,newname) ; }
  virtual ~RooGExpModel();

  virtual Int_t basisCode(const char* name) const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

  void advertiseFlatScaleFactorIntegral(Bool_t flag) { _flatSFInt = flag ; }

  void advertiseAsymptoticIntegral(Bool_t flag) { _asympInt = flag ; }  // added FMV,07/24/03

protected:
  virtual Double_t evaluate() const ;

private:
  //Double_t calcDecayConv(Double_t sign, Double_t tau, Double_t sig, Double_t rtau) const ;
  Double_t calcDecayConv(Double_t sign, Double_t tau, Double_t sig, Double_t rtau, Double_t fsign) const ;
   // modified FMV,08/13/03
  std::complex<Double_t> calcSinConv(Double_t sign, Double_t sig, Double_t tau, Double_t omega, Double_t rtau, Double_t fsign) const ;
  Double_t calcSinConv(Double_t sign, Double_t sig, Double_t tau, Double_t rtau, Double_t fsign) const ;
  std::complex<Double_t> calcSinConvNorm(Double_t sign, Double_t tau, Double_t omega,
                        Double_t sig, Double_t rtau, Double_t fsign, const char* rangeName) const ; // modified FMV,07/24/03
  Double_t calcSinConvNorm(Double_t sign, Double_t tau,
        Double_t sig, Double_t rtau, Double_t fsign, const char* rangeName) const ; // added FMV,08/18/03
  //Double_t calcSinhConv(Double_t sign, Double_t sign1, Double_t sign2, Double_t tau, Double_t dgamma, Double_t sig, Double_t rtau, Double_t fsign) const ;
  //Double_t calcCoshConv(Double_t sign, Double_t tau, Double_t dgamma, Double_t sig, Double_t rtau, Double_t fsign) const ;

  static std::complex<Double_t> evalCerfApprox(Double_t swt, Double_t u, Double_t c);

  // Calculate common normalization factors
  // added FMV,07/24/03
  std::complex<Double_t> evalCerfInt(Double_t sign, Double_t wt, Double_t tau, Double_t umin, Double_t umax, Double_t c) const ;
  Double_t evalCerfInt(Double_t sign, Double_t tau, Double_t umin, Double_t umax, Double_t c) const ;

  RooRealProxy _mean;
  RooRealProxy sigma ;
  RooRealProxy rlife ;
  RooRealProxy _meanSF;
  RooRealProxy ssf ;
  RooRealProxy rsf ;

  Bool_t _flip ;
  Bool_t _nlo ;
  Bool_t _flatSFInt ;
  Bool_t _asympInt ;  // added FMV,07/24/03

  ClassDef(RooGExpModel,2) // Gauss (x) Exponential resolution model
};

#endif
