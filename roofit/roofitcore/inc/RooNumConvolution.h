/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumConvolution.h,v 1.4 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_NUM_CONVOLUTION
#define ROO_NUM_CONVOLUTION

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooListProxy.h"
#include "RooNumIntConfig.h"

class RooConvIntegrandBinding ;
class RooAbsIntegrator ;
class TH2 ;

class RooNumConvolution : public RooAbsReal {
public:

  RooNumConvolution() ;

  RooNumConvolution(const char *name, const char *title,
            RooRealVar& convVar, RooAbsReal& pdf, RooAbsReal& resmodel, const RooNumConvolution* proto=0) ;

  RooNumConvolution(const RooNumConvolution& other, const char* name=0) ;

  TObject* clone(const char* newname) const override { return new RooNumConvolution(*this,newname) ; }
  ~RooNumConvolution() override ;

  double evaluate() const override ;

  RooNumIntConfig& convIntConfig() { _init = false ; return _convIntConfig ; }
  const RooNumIntConfig& convIntConfig() const { _init = false ; return _convIntConfig ; }

  void clearConvolutionWindow() ;
  void setConvolutionWindow(RooAbsReal& centerParam, RooAbsReal& widthParam, double widthScaleFactor=1) ;

  void setCallWarning(Int_t threshold=2000) ;
  void setCallProfiling(bool flag, Int_t nbinX = 40, Int_t nbinCall = 40, Int_t nCallHigh=1000) ;
  const TH2* profileData() const { return _doProf ? _callHist : 0 ; }

  // Access components
  RooRealVar&  var() const { return (RooRealVar&) _origVar.arg() ; }
  RooAbsReal&  pdf() const { return (RooAbsReal&) _origPdf.arg() ; }
  RooAbsReal&  model() const { return (RooAbsReal&) _origModel.arg() ; }

protected:

  friend class RooNumConvPdf ;

  mutable bool _init ;
  void initialize() const ;
  bool redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive) override ;

  void printCompactTreeHook(std::ostream& os, const char* indent="") override ;

  RooNumIntConfig _convIntConfig ; ///< Configuration of numeric convolution integral ;
  mutable RooConvIntegrandBinding* _integrand ; ///<! Binding of Convolution Integrand function
  mutable RooAbsIntegrator* _integrator ;  ///<! Numeric integrator of convolution integrand

  RooRealProxy _origVar ;         ///< Original convolution variable
  RooRealProxy _origPdf ;         ///< Original input PDF
  RooRealProxy _origModel ;       ///< Original resolution model

  mutable RooArgSet    _ownedClonedPdfSet ;   ///< Owning set of cloned PDF components
  mutable RooArgSet    _ownedClonedModelSet ; ///< Owning set of cloned model components

  mutable RooAbsReal*  _cloneVar ;        ///< Pointer to cloned convolution variable
  mutable RooAbsReal*  _clonePdf ;        ///< Pointer to cloned PDF
  mutable RooAbsReal*  _cloneModel ;      ///< Pointer to cloned model

  friend class RooConvGenContext ;
  RooRealVar&  cloneVar()   const { if (!_init) initialize() ; return (RooRealVar&) *_cloneVar ; }
  RooAbsReal&   clonePdf()   const { if (!_init) initialize() ; return (RooAbsReal&)  *_clonePdf ; }
  RooAbsReal&   cloneModel() const { if (!_init) initialize() ; return (RooAbsReal&)  *_cloneModel ; }

  bool       _useWindow   ;     ///< Switch to activate window convolution
  double     _windowScale ;     ///< Scale factor for window parameter
  RooListProxy _windowParam ;     ///< Holder for optional convolution integration window scaling parameter

  Int_t        _verboseThresh ;   ///< Call count threshold for verbose printing
  bool       _doProf   ;        ///< Switch to activate profiling option
  TH2*         _callHist ;        ///<! Histogram recording number of calls per convolution integral calculation

  ClassDefOverride(RooNumConvolution,1)   // Operator PDF implementing numeric convolution of 2 input functions
};

#endif
