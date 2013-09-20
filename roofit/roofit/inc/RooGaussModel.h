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

#include <cmath>
#include <complex>

#include "RooResolutionModel.h"
#include "RooRealProxy.h"
#include "RooMath.h"

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
  inline RooGaussModel() : _flatSFInt(kFALSE), _asympInt(kFALSE) { }
  RooGaussModel(const char *name, const char *title, RooRealVar& x, 
		RooAbsReal& mean, RooAbsReal& sigma) ; 
  RooGaussModel(const char *name, const char *title, RooRealVar& x, 
		RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& msSF) ; 
  RooGaussModel(const char *name, const char *title, RooRealVar& x, 
		RooAbsReal& mean, RooAbsReal& sigma, RooAbsReal& meanSF, RooAbsReal& sigmaSF) ; 
  RooGaussModel(const RooGaussModel& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooGaussModel(*this,newname) ; }
  virtual ~RooGaussModel();
  
  virtual Int_t basisCode(const char* name) const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual Double_t analyticalIntegral(Int_t code, const char* rangeName) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

  void advertiseFlatScaleFactorIntegral(Bool_t flag) { _flatSFInt = flag ; }

  void advertiseAymptoticIntegral(Bool_t flag) { _asympInt = flag ; }  // added FMV,07/24/03

protected:

  virtual Double_t evaluate() const ;
  static std::complex<Double_t> evalCerfApprox(Double_t swt, Double_t u, Double_t c);

  // Calculate exp(-u^2) cwerf(swt*c + i(u+c)), taking care of numerical instabilities
  static inline std::complex<Double_t> evalCerf(Double_t swt, Double_t u, Double_t c)
  {
    std::complex<Double_t> z(swt*c,u+c);
    return (z.imag()>-4.0) ? (std::exp(-u*u)*RooMath::faddeeva_fast(z)) : evalCerfApprox(swt,u,c);
  }
    
  // Calculate common normalization factors 
  std::complex<Double_t> evalCerfInt(Double_t sign, Double_t wt, Double_t tau, Double_t umin, Double_t umax, Double_t c) const;

  Bool_t _flatSFInt ;

  Bool_t _asympInt ;  // added FMV,07/24/03
  
  RooRealProxy mean ;
  RooRealProxy sigma ;
  RooRealProxy msf ;
  RooRealProxy ssf ;

  ClassDef(RooGaussModel,1) // Gaussian Resolution Model
};

#endif









