/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumConvPdf.h,v 1.2 2007/05/11 10:42:36 verkerke Exp $
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

  RooNumConvPdf() ;

  RooNumConvPdf(const char *name, const char *title,
                RooRealVar& convVar, RooAbsPdf& pdf, RooAbsPdf& resmodel) ;

  RooNumConvPdf(const RooNumConvPdf& other, const char* name=nullptr) ;

  TObject* clone(const char* newname) const override { return new RooNumConvPdf(*this,newname) ; }
  ~RooNumConvPdf() override ;

  double evaluate() const override ;

  // Calls forwarded to RooNumConvolution
  inline RooNumIntConfig& convIntConfig() { return conv().convIntConfig() ; }
  inline void clearConvolutionWindow() { conv().clearConvolutionWindow() ; }
  inline void setConvolutionWindow(RooAbsReal& centerParam, RooAbsReal& widthParam, double widthScaleFactor=1)
   { conv().setConvolutionWindow(centerParam,widthParam,widthScaleFactor) ; }
  inline void setCallWarning(Int_t threshold=2000) { conv().setCallWarning(threshold) ; }
  inline void setCallProfiling(bool flag, Int_t nbinX = 40, Int_t nbinCall = 40, Int_t nCallHigh=1000)
   { conv().setCallProfiling(flag,nbinX,nbinCall,nCallHigh) ; }
  inline const TH2* profileData() const { return conv().profileData() ; }

  // Access components
  RooRealVar&  var() const { return (RooRealVar&)(const_cast<RooAbsReal&>(_origVar.arg())) ; }
  RooAbsReal&  pdf() const { return const_cast<RooAbsReal&>(_origPdf.arg()) ; }
  RooAbsReal&  model() const { return const_cast<RooAbsReal&>(_origModel.arg()) ; }

  void printMetaArgs(std::ostream& os) const override ;

protected:

  // WVE Store all properties of RooNumConvolution here so that can be take
  // along in the copy ctor.

  RooNumConvolution& conv() const { if (!_init) initialize() ; return *_conv ; }

  mutable bool _init = false; ///<! do not persist
  void initialize() const ;
  mutable std::unique_ptr<RooNumConvolution> _conv ; ///<! Actual convolution calculation

  RooRealProxy _origVar ;         ///< Original convolution variable
  RooRealProxy _origPdf ;         ///< Original input PDF
  RooRealProxy _origModel ;       ///< Original resolution model

  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                       const RooArgSet* auxProto=nullptr, bool verbose= false) const override ;

  friend class RooConvGenContext ;

  ClassDefOverride(RooNumConvPdf,1)     // Operator PDF implementing numeric convolution of 2 input PDFs
};

#endif
