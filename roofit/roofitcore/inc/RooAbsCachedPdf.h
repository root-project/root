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
#include "RooAICRegistry.h"
#include "RooChangeTracker.h"

#include <map>

class RooAbsCachedPdf : public RooAbsPdf {
public:

  // Default constructor
  RooAbsCachedPdf() : _cacheMgr(this,10) {}
  RooAbsCachedPdf(const char *name, const char *title, Int_t ipOrder=0);
  RooAbsCachedPdf(const RooAbsCachedPdf& other, const char* name=0) ;

  virtual Double_t getValV(const RooArgSet* set=0) const ;
  virtual Bool_t selfNormalized() const { 
    // Declare p.d.f self normalized
    return kTRUE ; 
  }

  RooAbsPdf* getCachePdf(const RooArgSet& nset) const {
    // Return RooHistPdf that represents cache histogram
    return getCachePdf(&nset) ;
  }
  RooDataHist* getCacheHist(const RooArgSet& nset) const {
    // Return RooDataHist with cached values
    return getCacheHist(&nset) ;
  }
  RooAbsPdf* getCachePdf(const RooArgSet* nset=0) const ;
  RooDataHist* getCacheHist(const RooArgSet* nset=0) const ;

  void setInterpolationOrder(Int_t order) ;
  Int_t getInterpolationOrder() const { 
    // Set interpolation order in RooHistPdf that represent cached histogram
    return _ipOrder ; 
  }

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const ;
  virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const ; 
  virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;


  class PdfCacheElem : public RooAbsCacheElement {
  public:
    PdfCacheElem(const RooAbsCachedPdf& self, const RooArgSet* nset) ;

    // Cache management functions
    virtual RooArgList containedArgs(Action) ;
    virtual void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) ;

    RooHistPdf* pdf() { return _pdf.get() ; }
    RooDataHist* hist() { return _hist.get() ; }
    const RooArgSet& nset() { return _nset ; }
    RooChangeTracker* paramTracker() { return _paramTracker.get() ; }

  private:
    // Payload
    std::unique_ptr<RooHistPdf>  _pdf ;
    std::unique_ptr<RooChangeTracker> _paramTracker ;
    std::unique_ptr<RooDataHist> _hist ;
    RooArgSet    _nset ;
    std::unique_ptr<RooAbsReal>  _norm ;

  } ;

  protected:
   
  PdfCacheElem* getCache(const RooArgSet* nset, Bool_t recalculate=kTRUE) const ;
  void clearCacheObject(PdfCacheElem& cache) const ;

  virtual const char* payloadUniqueSuffix() const { return 0 ; }
  
  friend class PdfCacheElem ;
  virtual const char* binningName() const { 
    // Return name of binning to be used for creation of cache histogram
    return "cache" ; 
  }
  virtual PdfCacheElem* createCache(const RooArgSet* nset) const { 
    // Create cache storage element
    return new PdfCacheElem(*this,nset) ; 
  }
  virtual const char* inputBaseName() const = 0 ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const = 0 ;
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const = 0 ;
  virtual RooAbsArg& pdfObservable(RooAbsArg& histObservable) const { return histObservable ; }
  virtual void fillCacheObject(PdfCacheElem& cache) const = 0 ;

  mutable RooObjCacheManager _cacheMgr ; //! The cache manager  
  Int_t _ipOrder ; // Interpolation order for cache histograms 
 
  TString cacheNameSuffix(const RooArgSet& nset) const ;
  virtual TString histNameSuffix() const { return TString("") ; }
  void disableCache(Bool_t flag) { 
    // Flag to disable caching mechanism
    _disableCache = flag ; 
  }

  mutable RooAICRegistry _anaReg ; //! Registry for analytical integration codes
  class AnaIntConfig {
  public:
    RooArgSet _allVars ;
    RooArgSet _anaVars ;
    const RooArgSet* _nset ;
    Bool_t    _unitNorm ;
  } ;
  mutable std::map<Int_t,AnaIntConfig> _anaIntMap ; //! Map for analytical integration codes



private:

  Bool_t _disableCache ; // Flag to run object in passthrough (= non-caching mode)

  ClassDef(RooAbsCachedPdf,2) // Abstract base class for cached p.d.f.s
};
 
#endif
