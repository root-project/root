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

#ifndef ROOCACHEDREAL
#define ROOCACHEDREAL

#include "RooAbsCachedReal.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"


class RooCachedReal : public RooAbsCachedReal {
public:
  RooCachedReal() : _cacheSource(false) {
    // coverity[UNINIT_CTOR]
  }
  RooCachedReal(const char *name, const char *title, RooAbsReal& _func, const RooArgSet& cacheObs);
  RooCachedReal(const char *name, const char *title, RooAbsReal& _func);
  RooCachedReal(const RooCachedReal& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooCachedReal(*this,newname); }
  ~RooCachedReal() override ;

  /// If flag is true the RooHistFunc that represent the cache histogram
  /// will use special boundary conditions for use with cumulative distribution
  /// functions: at the lower bound the function is forced to converge at zero and the upper
  /// bound is the function is forced to converge at 1.0
  void setCdfBoundaries(bool flag) {
    _useCdfBoundaries = flag ;
  }
  /// If true the c.d.f boundary mode is active
  bool getCdfBoundaries() const {
    return _useCdfBoundaries ;
  }

  bool cacheSource() const { return _cacheSource ; }
  void setCacheSource(bool flag) { _cacheSource = flag ; }

protected:

  /// Return base name for caches, i.e. the name of the cached function
  const char* inputBaseName() const override {
    return func.arg().GetName() ;
  } ;
  RooArgSet* actualObservables(const RooArgSet& nset) const override ;
  RooArgSet* actualParameters(const RooArgSet& nset) const override ;
  void fillCacheObject(FuncCacheElem& cacheFunc) const override ;
  /// Dummy evaluate, it is never called
  Double_t evaluate() const override {
    return func ;
  }

  void operModeHook() override ;

  FuncCacheElem* createCache(const RooArgSet* nset) const override ;

  const char* payloadUniqueSuffix() const override { return func.arg().aggregateCacheUniqueSuffix() ; }

  RooRealProxy func ;           ///< Proxy to function being cached
  RooSetProxy  _cacheObs ;      ///< Variables to be cached
  bool _useCdfBoundaries ;    ///< Are c.d.f boundary conditions used by the RooHistFuncs?
  bool _cacheSource ;         ///< Keep an attached clone of the source in the cache for fast operation

private:

  ClassDefOverride(RooCachedReal,2) // P.d.f class that wraps another p.d.f and caches its output

};

#endif
