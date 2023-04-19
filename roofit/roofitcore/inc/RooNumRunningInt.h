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

#ifndef ROONUMRUNNINGINT
#define ROONUMRUNNINGINT

#include "RooAbsCachedReal.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"

#include <string>
#include <vector>

class RooNumRunningInt : public RooAbsCachedReal {
public:
  RooNumRunningInt(const char *name, const char *title, RooAbsReal& _func, RooRealVar& _x, const char* binningName="cache");
  RooNumRunningInt(const RooNumRunningInt& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooNumRunningInt(*this,newname); }
  ~RooNumRunningInt() override ;

protected:

  class RICacheElem: public FuncCacheElem {
  public:
    RICacheElem(const RooNumRunningInt& ri, const RooArgSet* nset) ;
    RooArgList containedArgs(Action) override ;
    void calculate(bool cdfmode) ;
    void addRange(Int_t ixlo, Int_t ixhi, Int_t nbins) ;
    void addPoint(Int_t ix) ;

    RooNumRunningInt* _self ;
    std::vector<double> _ax ;
    std::vector<double> _ay ;
    RooRealVar* _xx ;

  } ;

  friend class RICacheElem ;
  const char* binningName() const override { return _binningName.c_str() ; }
  FuncCacheElem* createCache(const RooArgSet* nset) const override ;
  const char* inputBaseName() const override ;
  RooArgSet* actualObservables(const RooArgSet& nset) const override ;
  RooFit::OwningPtr<RooArgSet> actualParameters(const RooArgSet& nset) const override ;
  void fillCacheObject(FuncCacheElem& cacheFunc) const override ;
  double evaluate() const override ;

  const char* payloadUniqueSuffix() const override { return func.arg().aggregateCacheUniqueSuffix() ; }

  RooRealProxy func ; ///< Proxy to functions whose running integral is calculated
  RooRealProxy x   ; ///< Intergrated observable
  std::string _binningName ; ///< Name of binning to be used for cache histogram

private:

  ClassDefOverride(RooNumRunningInt,1) // Numeric calculator for running integral of a given function

};

#endif
