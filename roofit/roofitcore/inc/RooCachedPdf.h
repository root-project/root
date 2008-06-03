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
  RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf, const RooArgSet& cacheObs);
  RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf);
  RooCachedPdf(const RooCachedPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooCachedPdf(*this,newname); }
  virtual ~RooCachedPdf() ;

  virtual void preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const ;

protected:

  virtual const char* inputBaseName() const { return pdf.arg().GetName() ; } ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const ;
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const ;
  virtual void fillCacheObject(PdfCacheElem& cachePdf) const ;
  virtual Double_t evaluate() const { return 0 ; } // dummy
  
  RooRealProxy pdf ; // Proxy to p.d.f being cached
  RooSetProxy  _cacheObs ; // Observable to be cached

private:

  ClassDef(RooCachedPdf,1) // P.d.f class that wraps another p.d.f and caches its output 

};
 
#endif
