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

#ifndef ROOABSSELFCACHEDPDF
#define ROOABSSELFCACHEDPDF

#include "RooAbsCachedPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooHistPdf.h"


class RooAbsSelfCachedPdf : public RooAbsCachedPdf {
public:

  RooAbsSelfCachedPdf() {} ;
  RooAbsSelfCachedPdf(const char *name, const char *title, Int_t ipOrder=0);
  RooAbsSelfCachedPdf(const RooAbsSelfCachedPdf& other, const char* name=nullptr) ;
  ~RooAbsSelfCachedPdf() override ;

protected:

  const char* inputBaseName() const override {
    // Use own name as base name for caches
    return GetName() ;
  }
  RooArgSet* actualObservables(const RooArgSet& nset) const override ;
  RooArgSet* actualParameters(const RooArgSet& nset) const override ;
  void fillCacheObject(PdfCacheElem& cache) const override ;

private:

  ClassDefOverride(RooAbsSelfCachedPdf,0) // Abstract base class for self-caching p.d.f.s
};

#endif
