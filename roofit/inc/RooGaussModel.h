/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGaussModel.rdl,v 1.11 2002/03/27 08:07:06 mwilson Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_GAUSS_MODEL
#define ROO_GAUSS_MODEL

#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooComplex.hh"
#include "RooFitCore/RooMath.hh"

class RooGaussModel : public RooResolutionModel {
public:

  enum RooGaussBasis { noBasis=0, expBasisMinus= 1, expBasisSum= 2, expBasisPlus= 3,
                                  sinBasisMinus=11, sinBasisSum=12, sinBasisPlus=13,
                                  cosBasisMinus=21, cosBasisSum=22, cosBasisPlus=23,
                                                                    linBasisPlus=33,
                                                                   quadBasisPlus=43 } ;
  enum BasisType { none=0, expBasis=1, sinBasis=2, cosBasis=3,
		   linBasis=4, quadBasis=5 } ;
  enum BasisSign { Both=0, Plus=+1, Minus=-1 } ;

  // Constructors, assignment etc
  inline RooGaussModel() { }
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
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

protected:

  virtual Double_t evaluate() const ;
  RooComplex evalCerfApprox(Double_t swt, Double_t u, Double_t c) const ;

  // Calculate exp(-u^2) cwerf(swt*c + i(u+c)), taking care of numerical instabilities
  inline RooComplex evalCerf(Double_t swt, Double_t u, Double_t c) const {
    RooComplex z(swt*c,u+c);
    return (z.im()>-4.0) ? RooMath::FastComplexErrFunc(z)*exp(-u*u) : evalCerfApprox(swt,u,c) ;
  }
    
  // Calculate Re(exp(-u^2) cwerf(swt*c + i(u+c))), taking care of numerical instabilities
  inline Double_t evalCerfRe(Double_t swt, Double_t u, Double_t c) const {
    RooComplex z(swt*c,u+c);
    return (z.im()>-4.0) ? RooMath::FastComplexErrFuncRe(z)*exp(-u*u) : evalCerfApprox(swt,u,c).re() ;
  }
  
  // Calculate Im(exp(-u^2) cwerf(swt*c + i(u+c))), taking care of numerical instabilities
  inline Double_t evalCerfIm(Double_t swt, Double_t u, Double_t c) const {
    RooComplex z(swt*c,u+c);
    return (z.im()>-4.0) ? RooMath::FastComplexErrFuncIm(z)*exp(-u*u) : evalCerfApprox(swt,u,c).im() ;
  }
  
  
  RooRealProxy mean ;
  RooRealProxy sigma ;
  RooRealProxy msf ;
  RooRealProxy ssf ;

  ClassDef(RooGaussModel,1) // Gaussian Resolution Model
};

#endif
