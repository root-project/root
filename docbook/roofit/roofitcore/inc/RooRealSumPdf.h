/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealSumPdf.h,v 1.10 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_SUM_PDF
#define ROO_REAL_SUM_PDF

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"

class RooRealSumPdf : public RooAbsPdf {
public:

  RooRealSumPdf() ;
  RooRealSumPdf(const char *name, const char *title);
  RooRealSumPdf(const char *name, const char *title, const RooArgList& funcList, const RooArgList& coefList, Bool_t extended=kFALSE) ;
  RooRealSumPdf(const char *name, const char *title,
		   RooAbsReal& func1, RooAbsReal& func2, RooAbsReal& coef1) ;
  RooRealSumPdf(const RooRealSumPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooRealSumPdf(*this,newname) ; }
  virtual ~RooRealSumPdf() ;

  Double_t evaluate() const ;
  virtual Bool_t checkObservables(const RooArgSet* nset) const ;	

  virtual Bool_t forceAnalyticalInt(const RooAbsArg&) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;

  const RooArgList& funcList() const { return _funcList ; }
  const RooArgList& coefList() const { return _coefList ; }

  virtual ExtendMode extendMode() const ; 

  virtual Double_t expectedEvents(const RooArgSet* nset) const ;
  virtual Double_t expectedEvents(const RooArgSet& nset) const { 
    // Return expected number of events for extended likelihood calculation
    // which is the sum of all coefficients
    return expectedEvents(&nset) ; 
  }

  void printMetaArgs(ostream& os) const ;

protected:
  
  class CacheElem : public RooAbsCacheElement {
  public:
    CacheElem()  {} ;
    virtual ~CacheElem() {} ; 
    virtual RooArgList containedArgs(Action) { RooArgList ret(_funcIntList) ; ret.add(_funcNormList) ; return ret ; }
    RooArgList _funcIntList ;
    RooArgList _funcNormList ;
  } ;
  mutable RooObjCacheManager _normIntMgr ; // The integration cache manager

  Bool_t _haveLastCoef ;

  RooListProxy _funcList ;   //  List of component FUNCs
  RooListProxy _coefList ;  //  List of coefficients
  TIterator* _funcIter ;     //! Iterator over FUNC list
  TIterator* _coefIter ;    //! Iterator over coefficient list
  Bool_t _extended ;        // Allow use as extended p.d.f.
  
private:

  ClassDef(RooRealSumPdf,2) // PDF constructed from a sum of (non-pdf) functions
};

#endif
