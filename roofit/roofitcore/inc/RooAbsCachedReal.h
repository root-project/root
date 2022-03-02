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

  RooAbsCachedReal() : _cacheMgr(this,10) {}
  RooAbsCachedReal(const char *name, const char *title, Int_t ipOrder=0);
  RooAbsCachedReal(const RooAbsCachedReal& other, const char* name=0) ;
  ~RooAbsCachedReal() override ;

  Double_t getValV(const RooArgSet* set=0) const override ;
  virtual Bool_t selfNormalized() const {
    // Declares function self normalized
    return kTRUE ;
  }

  void setInterpolationOrder(Int_t order) ;
  Int_t getInterpolationOrder() const {
    // Set interpolation order in RooHistFuncs that represent cache histograms
    return _ipOrder ;
  }

  Bool_t forceAnalyticalInt(const RooAbsArg& /*dep*/) const override {
    // Force all observables to be offered for internal integration
    return kTRUE ;
  }

  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const override ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override ;

  void disableCache(Bool_t flag) {
    // Switch to disable caching mechanism
    _disableCache = flag ;
  }

protected:

  class FuncCacheElem : public RooAbsCacheElement {
  public:
    FuncCacheElem(const RooAbsCachedReal& self, const RooArgSet* nset) ;
    ~FuncCacheElem() override ;

    // Cache management functions
    RooArgList containedArgs(Action) override ;
    void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) override ;

    RooHistFunc* func() { return _func ; }
    RooDataHist* hist() { return _hist ; }
    RooChangeTracker* paramTracker() { return _paramTracker ; }

    RooAbsReal* sourceClone() { return _sourceClone ; }
    void setSourceClone(RooAbsReal* newSource) { delete _sourceClone ; _sourceClone = newSource ; }

    Bool_t cacheSource() { return _cacheSource ; }
    void setCacheSource(Bool_t flag) { _cacheSource = flag ; }

  private:
    // Payload
    RooHistFunc*      _func ;
    RooChangeTracker* _paramTracker ;
    RooDataHist*      _hist ;
    RooAbsReal*       _sourceClone ;
    Bool_t            _cacheSource ;
  } ;

  FuncCacheElem* getCache(const RooArgSet* nset) const ;
  void clearCacheObject(FuncCacheElem& cache) const ;

  virtual const char* payloadUniqueSuffix() const { return 0 ; }

  friend class FuncCacheElem ;
  virtual const char* binningName() const {
    // Returns name of binning to be used for cache histogram creation
    return "cache" ;
  }
  virtual FuncCacheElem* createCache(const RooArgSet* nset) const ;
  virtual const char* inputBaseName() const = 0 ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const = 0 ;
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const = 0 ;
  virtual void fillCacheObject(FuncCacheElem& cache) const = 0 ;

  mutable RooObjCacheManager _cacheMgr ; ///<! The cache manager


  Int_t _ipOrder ; ///< Interpolation order for cache histograms

  TString cacheNameSuffix(const RooArgSet& nset) const ;

  mutable std::map<Int_t,std::pair<const RooArgSet*,const RooArgSet*> > _anaIntMap ; ///<! Map for analytical integration codes


private:

  Bool_t _disableCache ; // Flag to run object in passthrough (= non-caching mode)

  ClassDefOverride(RooAbsCachedReal,1) // Abstract base class for cached p.d.f.s
};

#endif
