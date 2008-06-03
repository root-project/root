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
  RooCachedReal(const char *name, const char *title, RooAbsReal& _func);
  RooCachedReal(const RooCachedReal& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooCachedReal(*this,newname); }
  virtual ~RooCachedReal() ;

  void setCdfBoundaries(Bool_t flag) { _useCdfBoundaries = flag ; }
  Bool_t getCdfBoundaries() const { return _useCdfBoundaries ; }

protected:

  virtual const char* inputBaseName() const { return func.arg().GetName() ; } ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const { return func.arg().getObservables(nset) ; }
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const { return func.arg().getParameters(nset) ; }
  virtual void fillCacheObject(FuncCacheElem& cacheFunc) const ;
  virtual Double_t evaluate() const { return 0 ; } // dummy
  
  RooRealProxy func ; // Proxy to function being cached
  Bool_t _useCdfBoundaries ;

private:

  ClassDef(RooCachedReal,1) // P.d.f class that wraps another p.d.f and caches its output 

};
 
#endif
