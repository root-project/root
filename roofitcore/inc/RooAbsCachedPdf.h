/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOABSCACHEDPDF
#define ROOABSCACHEDPDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooHistPdf.h"
#include "RooObjCacheManager.h"
#include <map>
class RooArgSet ;
 
class RooAbsCachedPdf : public RooAbsPdf {
public:

  RooAbsCachedPdf() {} ;
  RooAbsCachedPdf(const char *name, const char *title, Int_t ipOrder=0);
  RooAbsCachedPdf(const RooAbsCachedPdf& other, const char* name=0) ;
  virtual ~RooAbsCachedPdf() ;

  virtual Double_t getVal(const RooArgSet* set=0) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  void setInterpolationOrder(Int_t order) ;
  Int_t getInterpolationOrder() const { return _ipOrder ; }

   virtual Bool_t forceAnalyticalInt(const RooAbsArg& /*dep*/) const { return kTRUE ; } 
   virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const ; 
   virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;

protected:

  class CacheElem : public RooAbsCacheElement {
  public:
    virtual ~CacheElem() {} ;
    // Payload
    RooHistPdf*  _pdf ;
    RooAbsReal* _params ;
    RooDataHist* _hist ;
    // Cache management functions
    virtual RooArgList containedArgs(Action) ;
    virtual void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) ;
  } ;

  const CacheElem* getCache(const RooArgSet* nset) const ;
  void clearCacheObject(CacheElem& cache) const ;

  virtual const char* inputBaseName() const = 0 ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const = 0 ;
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const = 0 ;
  virtual void fillCacheObject(CacheElem& cache) const = 0 ;

  mutable RooObjCacheManager _cacheMgr ; // The cache manager

  
  Int_t _ipOrder ; // Interpolation order for cache histograms 
 
  TString cacheNameSuffix(const RooArgSet& nset) const ;
  void disableCache(Bool_t flag) { _disableCache = flag ; }

  mutable std::map<Int_t,std::pair<const RooArgSet*,const RooArgSet*> > _anaIntMap ; //!
  

private:

  Bool_t _disableCache ; // Flag to run object in passthrough (= non-caching mode)

  ClassDef(RooAbsCachedPdf,0) // Abstract base class for cached p.d.f.s
};
 
#endif
