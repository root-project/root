/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOLINEARMORPH
#define ROOLINEARMORPH

#include "RooAbsCachedPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include <vector>

class RooBrentRootFinder;

class TH1D;

class RooIntegralMorph : public RooAbsCachedPdf {
public:
   RooIntegralMorph() :  _cache(nullptr)  {
    // coverity[UNINIT_CTOR]
  } ;
  RooIntegralMorph(const char *name, const char *title,
         RooAbsReal& _pdf1,
         RooAbsReal& _pdf2,
           RooAbsReal& _x,
         RooAbsReal& _alpha, bool cacheAlpha=false);
  RooIntegralMorph(const RooIntegralMorph& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooIntegralMorph(*this,newname); }
  inline ~RooIntegralMorph() override { }

  bool selfNormalized() const override {
    // P.d.f is self normalized
    return true ;
  }
  void setCacheAlpha(bool flag) {
    // Activate caching of p.d.f. shape for all values of alpha as well
    _cacheMgr.sterilize() ; _cacheAlpha = flag ;
  }
  bool cacheAlpha() const {
    // If true caching of p.d.f for all alpha values is active
    return _cacheAlpha ;
  }

  void preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const override ;

  class MorphCacheElem : public RooAbsCachedPdf::PdfCacheElem {
  public:
    MorphCacheElem(RooIntegralMorph& self, const RooArgSet* nset) ;
    ~MorphCacheElem() override ;
    void calculate(TIterator* iter) ;
    RooArgList containedArgs(Action) override ;

  protected:

    void findRange() ;
    double calcX(double y, bool& ok) ;
    Int_t binX(double x) ;
    void fillGap(Int_t ixlo, Int_t ixhi,double splitPoint=0.5) ;
    void interpolateGap(Int_t ixlo, Int_t ixhi) ;

    RooIntegralMorph* _self ; //
    std::unique_ptr<RooArgSet> _nset ;
    RooAbsPdf* _pdf1 ; // PDF1
    RooAbsPdf* _pdf2 ; // PDF2
    RooRealVar* _x   ; // X
    RooAbsReal* _alpha ; // ALPHA
    std::unique_ptr<RooAbsReal> _c1 ; // CDF of PDF 1
    std::unique_ptr<RooAbsReal> _c2 ; // CDF of PDF 2
    RooAbsFunc* _cb1 ; // Binding of CDF1
    RooAbsFunc* _cb2 ; // Binding of CDF2
    std::unique_ptr<RooBrentRootFinder> _rf1; // ROOT finder on CDF1
    std::unique_ptr<RooBrentRootFinder> _rf2; // ROOT finder of CDF2 ;

    std::vector<double> _yatX ; //
    std::vector<double> _calcX; //
    Int_t _yatXmin, _yatXmax ;
    Int_t _ccounter ;

    double _ycutoff ;

  } ;

protected:

  friend class MorphCacheElem ;
  PdfCacheElem* createCache(const RooArgSet* nset) const override ;
  const char* inputBaseName() const override ;
  RooArgSet* actualObservables(const RooArgSet& nset) const override ;
  RooFit::OwningPtr<RooArgSet> actualParameters(const RooArgSet& nset) const override ;
  void fillCacheObject(PdfCacheElem& cache) const override ;

  RooRealProxy pdf1 ; // First input shape
  RooRealProxy pdf2 ; // Second input shape
  RooRealProxy x ;    // Observable
  RooRealProxy alpha ; // Interpolation parameter
  bool _cacheAlpha ; // If true, both (x,alpha) are cached
  mutable MorphCacheElem* _cache ; // Current morph cache element in use


  double evaluate() const override ;

private:

  ClassDefOverride(RooIntegralMorph,1) // Linear shape interpolation operator p.d.f
};

#endif
