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


#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooDataSet.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"
#include "RooAbsCacheElement.h"

class RooResolutionModel ;
class RooRealVar ;
class RooConvGenContext ;

class RooAbsAnaConvPdf : public RooAbsPdf {
public:

  // Constructors, assignment etc
  RooAbsAnaConvPdf() ;
  RooAbsAnaConvPdf(const char *name, const char *title,
         const RooResolutionModel& model,
         RooRealVar& convVar) ;

  RooAbsAnaConvPdf(const RooAbsAnaConvPdf& other, const char* name=0);
  ~RooAbsAnaConvPdf() override;

  Int_t declareBasis(const char* expression, const RooArgList& params) ;
  void printMultiline(std::ostream& stream, Int_t contents, bool verbose=false, TString indent= "") const override ;

  // Coefficient normalization access
  inline double getCoefNorm(Int_t coefIdx, const RooArgSet& nset, const char* rangeName) const {
    // Returns normalization integral for coefficient coefIdx for observables nset in range rangeNae
    return getCoefNorm(coefIdx,&nset,rangeName) ;
  }
  double getCoefNorm(Int_t coefIdx, const RooArgSet* nset=0, const char* rangeName=0) const {
       return getCoefNorm(coefIdx,nset,RooNameReg::ptr(rangeName));
  }

  // Analytical integration support
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override ;

  // Coefficient Analytical integration support
  virtual Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual double coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=0) const ;
  bool forceAnalyticalInt(const RooAbsArg& dep) const override ;

  virtual double coefficient(Int_t basisIndex) const = 0 ;
  virtual RooArgSet* coefVars(Int_t coefIdx) const ;

  bool isDirectGenSafe(const RooAbsArg& arg) const override ;

  void setCacheAndTrackHints(RooArgSet&) override ;

  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0,
                                       const RooArgSet* auxProto=0, bool verbose= false) const override ;
  virtual bool changeModel(const RooResolutionModel& newModel) ;

  /// Retrieve the convolution variable.
  RooAbsRealLValue* convVar();
  /// Retrieve the convolution variable.
  const RooAbsRealLValue* convVar() const {
    return const_cast<RooAbsAnaConvPdf*>(this)->convVar();
  }

protected:
  double getCoefNorm(Int_t coefIdx, const RooArgSet* nset, const TNamed* rangeName) const ;

  bool _isCopy ;

  double evaluate() const override ;

  void makeCoefVarList(RooArgList&) const ;

  friend class RooConvGenContext ;

  RooRealProxy _model ;   ///< Original model
  RooRealProxy _convVar ; ///< Convolution variable

  RooArgSet* parseIntegrationRequest(const RooArgSet& intSet, Int_t& coefCode, RooArgSet* analVars=0) const ;

  RooListProxy _convSet  ;  ///<  Set of (resModel (x) basisFunc) convolution objects
  RooArgList _basisList ;   ///<!  List of created basis functions


  class CacheElem : public RooAbsCacheElement {
  public:
    ~CacheElem() override {} ;

    RooArgList containedArgs(Action) override {
      RooArgList l(_coefVarList) ;
      l.add(_normList) ;
      return l ;
    }

    RooArgList _coefVarList ;
    RooArgList _normList ;
  } ;
  mutable RooObjCacheManager _coefNormMgr ; ///<! Coefficient normalization manager

  mutable RooAICRegistry _codeReg ;         ///<! Registry of analytical integration codes

  ClassDefOverride(RooAbsAnaConvPdf,3) // Abstract Composite Convoluted PDF
};

#endif
