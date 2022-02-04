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

#ifndef ROOCACHEDPDF
#define ROOCACHEDPDF

#include "RooAbsCachedPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"


class RooCachedPdf : public RooAbsCachedPdf {
public:
  RooCachedPdf() {} ;
  RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf, const RooArgSet& cacheObs);
  RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf);
  RooCachedPdf(const RooCachedPdf& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooCachedPdf(*this,newname); }
  ~RooCachedPdf() override ;

  void preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const override ;

protected:

  /// Return the base name for cache objects, in this case the name of the cached p.d.f
  const char* inputBaseName() const override {
    return pdf.arg().GetName() ;
  } ;
  RooArgSet* actualObservables(const RooArgSet& nset) const override ;
  RooArgSet* actualParameters(const RooArgSet& nset) const override ;
  void fillCacheObject(PdfCacheElem& cachePdf) const override ;
  Double_t evaluate() const override {
    // Dummy evaluate, it is never called
    return 0 ;
  }

  const char* payloadUniqueSuffix() const override { return pdf.arg().aggregateCacheUniqueSuffix() ; }

  RooRealProxy pdf ;       ///< Proxy to p.d.f being cached
  RooSetProxy  _cacheObs ; ///< Observable to be cached

private:

  ClassDefOverride(RooCachedPdf,1) // P.d.f class that wraps another p.d.f and caches its output

};

#endif
