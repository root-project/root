/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NUM_CONV_PDF
#define ROO_NUM_CONV_PDF

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooSetProxy.hh"
#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooNumIntConfig.hh"

class RooConvIntegrandBinding ;
class RooAbsIntegrator ;
class TH2 ;

class RooNumConvPdf : public RooAbsPdf {
public:

  RooNumConvPdf(const char *name, const char *title, 
	         RooRealVar& convVar, RooAbsPdf& pdf, RooAbsPdf& resmodel) ;

  RooNumConvPdf(const RooNumConvPdf& other, const char* name=0) ;

  virtual TObject* clone(const char* newname) const { return new RooNumConvPdf(*this,newname) ; }
  virtual ~RooNumConvPdf() ;

  Double_t evaluate() const ;

  RooNumIntConfig& convIntConfig() { _init = kFALSE ; return _convIntConfig ; }

  void clearConvolutionWindow() ;
  void setConvolutionWindow(RooAbsReal& centerParam, RooAbsReal& widthParam, Double_t widthScaleFactor=1) ;

  void setCallWarning(Int_t threshold=2000) ;
  void setCallProfiling(Bool_t flag, Int_t nbinX = 40, Int_t nbinCall = 40, Int_t nCallHigh=1000) ;
  const TH2* profileData() const { return _doProf ? _callHist : 0 ; }

protected:

  Bool_t _init ;
  void initialize(RooRealVar& convVar, RooAbsPdf& pdf, RooAbsPdf& model) ;
  Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  virtual void printCompactTreeHook(std::ostream& os, const char* indent="") ;

  RooNumIntConfig _convIntConfig ; // Configuration of numeric convolution integral ;
  RooConvIntegrandBinding* _integrand ; //! Binding of Convolution Integrand function
  RooAbsIntegrator* _integrator ;  //! Numeric integrator of convolution integrand

  RooRealProxy _origVar ;         // Original convolution variable
  RooRealProxy _origPdf ;         // Original input PDF
  RooRealProxy _origModel ;       // Original resolution model

  RooArgSet    _ownedClonedPdfSet ;   // Owning set of cloned PDF components
  RooArgSet    _ownedClonedModelSet ; // Owning set of cloned model components

  RooAbsReal*  _cloneVar ;        // Pointer to cloned convolution variable
  RooAbsReal*  _clonePdf ;        // Pointer to cloned PDF 
  RooAbsReal*  _cloneModel ;      // Pointer to cloned model

  Bool_t       _useWindow   ;     // Switch to activate window convolution
  Double_t     _windowScale ;     // Scale factor for window parameter
  RooListProxy _windowParam ;     // Holder for optional convolution integration window scaling parameter

  Int_t        _verboseThresh ;   // Call count threshold for verbose printing
  Bool_t       _doProf   ;        // Switch to activate profiling option
  TH2*         _callHist ;        //! Histogram recording number of calls per convolution integral calculation

  ClassDef(RooNumConvPdf,0)          // Operator PDF implementing numeric convolution of 2 input PDFs
};

#endif
