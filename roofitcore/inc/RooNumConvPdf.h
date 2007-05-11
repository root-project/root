/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumConvPdf.rdl,v 1.4 2005/02/25 14:23:00 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NUM_CONV_PDF
#define ROO_NUM_CONV_PDF

#include "RooAbsPdf.h"
#include "RooNumConvolution.h"

class TH2 ;
class RooArgSet ;
class RooDataSet ;

class RooNumConvPdf : public RooAbsPdf {
public:

  RooNumConvPdf(const char *name, const char *title, 
                RooRealVar& convVar, RooAbsPdf& pdf, RooAbsPdf& resmodel) ;

  RooNumConvPdf(const RooNumConvPdf& other, const char* name=0) ;

  virtual TObject* clone(const char* newname) const { return new RooNumConvPdf(*this,newname) ; }
  virtual ~RooNumConvPdf() ;

  virtual Double_t evaluate() const ;

  // Calls forwarded to RooNumConvolution
  inline RooNumIntConfig& convIntConfig() { return conv().convIntConfig() ; }
  inline void clearConvolutionWindow() { conv().clearConvolutionWindow() ; }
  inline void setConvolutionWindow(RooAbsReal& centerParam, RooAbsReal& widthParam, Double_t widthScaleFactor=1) 
	{ conv().setConvolutionWindow(centerParam,widthParam,widthScaleFactor) ; }
  inline void setCallWarning(Int_t threshold=2000) { conv().setCallWarning(threshold) ; }
  inline void setCallProfiling(Bool_t flag, Int_t nbinX = 40, Int_t nbinCall = 40, Int_t nCallHigh=1000) 
	{ conv().setCallProfiling(flag,nbinX,nbinCall,nCallHigh) ; }
  inline const TH2* profileData() const { return conv().profileData() ; }

  // Access components
  RooRealVar&  var() const { return (RooRealVar&) _origVar.arg() ; }
  RooAbsReal&  pdf() const { return (RooAbsReal&) _origPdf.arg() ; }
  RooAbsReal&  model() const { return (RooAbsReal&) _origModel.arg() ; }

protected:

  // WVE Store all properties of RooNumConvolution here so that can be take
  // along in the copy ctor.

  RooNumConvolution& conv() const { if (!_init) initialize() ; return *_conv ; }

  mutable Bool_t _init ;
  void initialize() const ;
  mutable RooNumConvolution* _conv ;

  RooRealProxy _origVar ;         // Original convolution variable
  RooRealProxy _origPdf ;         // Original input PDF
  RooRealProxy _origModel ;       // Original resolution model

  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0, 
                                       const RooArgSet* auxProto=0, Bool_t verbose= kFALSE) const ;

  friend class RooConvGenContext ;

  ClassDef(RooNumConvPdf,0)          // Operator PDF implementing numeric convolution of 2 input PDFs
};

#endif
