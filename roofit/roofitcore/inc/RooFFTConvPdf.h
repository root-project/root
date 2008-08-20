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

#ifndef ROOFFTCONVPDF
#define ROOFFTCONVPDF

#include "RooAbsCachedPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooHistPdf.h"
#include "TVirtualFFT.h"
class RooRealVar ;

#include <map>
 
class RooFFTConvPdf : public RooAbsCachedPdf {
public:

  RooFFTConvPdf() {} ;
  RooFFTConvPdf(const char *name, const char *title, RooRealVar& convVar, RooAbsPdf& pdf1, RooAbsPdf& pdf2, Int_t ipOrder=2);
  RooFFTConvPdf(const RooFFTConvPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooFFTConvPdf(*this,newname); }
  virtual ~RooFFTConvPdf() ;

  void setShift(Double_t val1, Double_t val2) { _shift1 = val1 ; _shift2 = val2 ; }

  Double_t bufferFraction() const { 
    // Return value of buffer fraction applied in FFT calculation array beyond either
    // end of the observable domain to reduce cyclical effects
    return _bufFrac ; 
  }
  void setBufferFraction(Double_t frac) ;

protected:

  RooRealProxy _x ;    // Convolution observable
  RooRealProxy _pdf1 ; // First input p.d.f
  RooRealProxy _pdf2 ; // Second input p.d.f

  Double_t*  scanPdf(RooRealVar& obs, RooAbsPdf& pdf, const RooDataHist& hist, const RooArgSet& slicePos, Int_t& N, Int_t& N2, Double_t shift) const ;

  class FFTCacheElem : public PdfCacheElem {
  public:
    FFTCacheElem(const RooFFTConvPdf& self, const RooArgSet* nset) ;
    ~FFTCacheElem() ;

    virtual RooArgList containedArgs(Action) ;

    TVirtualFFT* fftr2c1 ;
    TVirtualFFT* fftr2c2 ;
    TVirtualFFT* fftc2r ;

    RooAbsPdf* pdf1Clone ;
    RooAbsPdf* pdf2Clone ;

  };

  friend class FFTCacheElem ;  

  virtual Double_t evaluate() const { RooArgSet dummy(_x.arg()) ; return getVal(&dummy) ; } ; // dummy
  virtual const char* inputBaseName() const ;
  virtual RooArgSet* actualObservables(const RooArgSet& nset) const ;
  virtual RooArgSet* actualParameters(const RooArgSet& nset) const ;
  virtual void fillCacheObject(PdfCacheElem& cache) const ;
  void fillCacheSlice(FFTCacheElem& cache, const RooArgSet& slicePosition) const ;

  virtual PdfCacheElem* createCache(const RooArgSet* nset) const ;

  // mutable std::map<const RooHistPdf*,CacheAuxInfo*> _cacheAuxInfo ; //! Auxilary Cache information (do not persist)
  Double_t _bufFrac ; // Sampling buffer size as fraction of domain size 

  Double_t  _shift1 ; 
  Double_t  _shift2 ; 

  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0, 
                                       const RooArgSet* auxProto=0, Bool_t verbose= kFALSE) const ;

  friend class RooConvGenContext ;

private:

  ClassDef(RooFFTConvPdf,1) // Convolution operator p.d.f based on numeric Fourier transforms
};
 
#endif
