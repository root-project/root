/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsAnaConvPdf.h,v 1.8 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_ABS_ANA_CONV_PDF
#define ROO_ABS_ANA_CONV_PDF


class TIterator  ;
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooDataSet.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"
#include "RooAbsCacheElement.h"

class RooResolutionModel ;
class RooRealVar ;
class RooAbsGenContext ;
class RooConvGenContext ;

class RooAbsAnaConvPdf : public RooAbsPdf {
public:

  // Constructors, assignment etc
  RooAbsAnaConvPdf() ;
  RooAbsAnaConvPdf(const char *name, const char *title, 
		   const RooResolutionModel& model, 
		   RooRealVar& convVar) ;

  RooAbsAnaConvPdf(const RooAbsAnaConvPdf& other, const char* name=0);
  virtual ~RooAbsAnaConvPdf();

  Int_t declareBasis(const char* expression, const RooArgList& params) ;
  virtual void printMultiline(ostream& stream, Int_t contents, Bool_t verbose=kFALSE, TString indent= "") const ;

  // Coefficient normalization access
  inline Double_t getCoefNorm(Int_t coefIdx, const RooArgSet& nset, const char* rangeName) const { 
    // Returns normalization integral for coefficient coefIdx for observables nset in range rangeNae
    return getCoefNorm(coefIdx,&nset,rangeName) ; 
  }
  Double_t getCoefNorm(Int_t coefIdx, const RooArgSet* nset=0, const char* rangeName=0) const ;

  // Analytical integration support
  virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;
  
  // Coefficient Analytical integration support
  virtual Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=0) const ;
  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const ; 
  
  virtual Double_t coefficient(Int_t basisIndex) const = 0 ;
  virtual RooArgSet* coefVars(Int_t coefIdx) const ;

  virtual Bool_t isDirectGenSafe(const RooAbsArg& arg) const ;
    
protected:

  Bool_t _isCopy ;

  virtual Double_t evaluate() const ;

  void makeCoefVarList(RooArgList&) const ;

  friend class RooConvGenContext ;
  Bool_t changeModel(const RooResolutionModel& newModel) ;
  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0, 
                                       const RooArgSet* auxProto=0, Bool_t verbose= kFALSE) const ;

  // Following pointers are only used during 
  // construction and need not to be proxied
  RooResolutionModel* _model   ; //! Original resolution model
  RooRealVar* _convVar ;         //! Convolution variable

  RooArgSet* parseIntegrationRequest(const RooArgSet& intSet, Int_t& coefCode, RooArgSet* analVars=0) const ;

  const RooRealVar* convVar() const ;  //  Convolution variable 

  RooListProxy _convSet  ;             //  Set of (resModel (x) basisFunc) convolution objects
  RooArgList _basisList ;              //!  List of created basis functions
  mutable RooArgSet* _convNormSet ;    //!  Subset of last normalization that applies to convolutions
  mutable TIterator* _convSetIter ;    //! Iterator over _convNormSet


  class CacheElem : public RooAbsCacheElement {
  public:
    virtual ~CacheElem() {} ;

    RooArgList containedArgs(Action) { 
      RooArgList l(_coefVarList) ; 
      l.add(_normList) ; 
      return l ; 
    }

    RooArgList _coefVarList ;
    RooArgList _normList ;
  } ;
  mutable RooObjCacheManager _coefNormMgr ; // Coefficient normalization manager

  mutable RooAICRegistry _codeReg ;   //! Registry of analytical integration codes

  ClassDef(RooAbsAnaConvPdf,1) // Abstract Composite Convoluted PDF
};

#endif
