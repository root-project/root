/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_GEXP_MODEL
#define ROO_GEXP_MODEL

#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooComplex.hh"
#include "RooFitCore/RooMath.hh"

class RooGExpModel : public RooResolutionModel {
public:

  enum RooGaussBasis { noBasis=0, expBasisPlus=1, expBasisMinus=2,
                                  sinBasisPlus=3, sinBasisMinus=4,
                                  cosBasisPlus=5, cosBasisMinus=6 } ;
  enum Type { Normal, Flipped };

  // Constructors, assignment etc
  inline RooGExpModel() { }
  RooGExpModel(const char *name, const char *title, RooRealVar& x, 
	       RooAbsReal& sigma, RooAbsReal& rlife, 
	       RooAbsReal& sigmaSF, RooAbsReal& rlifeSF,
	       Bool_t nlo=kFALSE, Type type=Normal) ; 

  RooGExpModel(const RooGExpModel& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooGExpModel(*this,newname) ; }
  virtual ~RooGExpModel();
  
  virtual Int_t basisCode(const char* name) const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;

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
  

  RooRealProxy sigma ;
  RooRealProxy rlife ;
  RooRealProxy ssf ;
  RooRealProxy rsf ;
  Bool_t _flip ;
  Bool_t _nlo ;

  ClassDef(RooGExpModel,1) // GExp Resolution Model
};

#endif
