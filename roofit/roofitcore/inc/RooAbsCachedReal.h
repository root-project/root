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

#ifndef ROOABSCACHEDREAL
#define ROOABSCACHEDREAL

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooHistFunc.h"
#include "RooObjCacheManager.h"
#include <map>
class RooChangeTracker ;
class RooArgSet ;
 
class RooAbsCachedReal : public RooAbsReal {
public:

  RooAbsCachedReal() {} ;
  RooAbsCachedReal(const char *name, const char *title, Int_t ipOrder=0);
  RooAbsCachedReal(const RooAbsCachedReal& other, const char* name=0) ;
  virtual ~RooAbsCachedReal() ;

  virtual Double_t getVal(const RooArgSet* set=0) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  void setInterpolationOrder(Int_t order) ;
  Int_t getInterpolationOrder() const { return _ipOrder ; }

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& /*dep*/) const { return kTRUE ; } 
  virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const ; 
  virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;
   
protected:

  class FuncCacheElem : public RooAbsCacheElement {
  public:
    FuncCacheElem(const RooAbsCachedReal& self, const RooArgSet* nset) ;
    virtual ~FuncCacheElem() {} ;

    // Cache management functions
    virtual RooArgList containedArgs(Action) ;
    virtual void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) ;

    RooHistFunc* func() { return _func ; }
    RooDataHist* hist() { return _hist ; }
    RooChangeTracker* paramTracker() { return _paramTracker ; }

  private:
    // Payload
    RooHistFunc*  _func ;
    RooChangeTracker* _paramTracker ;
    RooDataHist* _hist ;
  } ;

  FuncCacheElem* getCache(const RooArgSet* nset) const ;
  void clearCacheObject(FuncCacheElem& cache) const ;

  virtual const char* binningName() const { return "cache" ; }
  virtual FuncCacheElem* createCache(const RooArgSet* nset) const ;
  virtual const char* inputBaseName() const = 0 ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const = 0 ;
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const = 0 ;
  virtual void fillCacheObject(FuncCacheElem& cache) const = 0 ;

  mutable RooObjCacheManager _cacheMgr ; // The cache manager

  
  Int_t _ipOrder ; // Interpolation order for cache histograms 
 
  TString cacheNameSuffix(const RooArgSet& nset) const ;
  void disableCache(Bool_t flag) { _disableCache = flag ; }

  mutable std::map<Int_t,std::pair<const RooArgSet*,const RooArgSet*> > _anaIntMap ; //!
  

private:

  Bool_t _disableCache ; // Flag to run object in passthrough (= non-caching mode)

  ClassDef(RooAbsCachedReal,0) // Abstract base class for cached p.d.f.s
};
 
#endif
