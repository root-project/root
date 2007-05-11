/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMinuit.rdl,v 1.13 2005/06/23 07:37:30 wverkerke Exp $
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
#ifndef ROO_MINUIT
#define ROO_MINUIT

#include "TObject.h"
#include "TStopwatch.h"
#include <fstream>

class RooAbsReal ;
class RooFitResult ;
class RooArgList ;
class RooRealVar ;
class RooArgSet ;
class TH2F ;

void RooMinuitGlue(Int_t& /*np*/, Double_t* /*gin*/,  Double_t &f, Double_t *par, Int_t /*flag*/) ;

class RooMinuit : public TObject {
public:

  RooMinuit(RooAbsReal& function) ;
  virtual ~RooMinuit() ;

  enum Strategy { Speed=0, Balance=1, Robustness=2 } ;
  enum PrintLevel { None=-1, Reduced=0, Normal=1, ExtraForProblem=2, Maximum=3 } ;
  void setStrategy(Int_t strat) ;
  void setErrorLevel(Double_t level) ;
  void setErrorHandling(Bool_t flag) { _handleLocalErrors = flag ; }
  void setEps(Double_t eps) ;
  void optimizeConst(Bool_t flag) ;

  RooFitResult* fit(const char* options) ;

  Int_t migrad() ;
  Int_t hesse() ;
  Int_t minos() ;
  Int_t minos(const RooArgSet& minosParamList) ;  // added FMV, 08/18/03
  Int_t seek() ;
  Int_t simplex() ;
  Int_t improve() ;

  RooFitResult* save(const char* name=0, const char* title=0) ;
  TH2F* contour(RooRealVar& var1, RooRealVar& var2, Double_t n1=1, Double_t n2=2, Double_t n3=0) ;

  Int_t setPrintLevel(Int_t newLevel) ; 
  Int_t setWarnLevel(Int_t newLevel) ;
  void setVerbose(Bool_t flag=kTRUE) { _verbose = flag ; }
  void setProfile(Bool_t flag=kTRUE) { _profile = flag ; }
  Bool_t setLogFile(const char* logfile=0) ;  
  
protected:

  friend void RooMinuitGlue(Int_t &np, Double_t *gin, Double_t &f, Double_t *par, Int_t flag) ;

  void profileStart() ;
  void profileStop() ;

  Bool_t synchronize(Bool_t verbose) ;  
  void backProp() ;

  inline Int_t getNPar() const { return _nPar ; }
  inline ofstream* logfile() const { return _logfile ; }
  inline Double_t& maxFCN() { return _maxFCN ; }

  Double_t getPdfParamVal(Int_t index) ;
  Double_t getPdfParamErr(Int_t index) ;	
  virtual Bool_t setPdfParamVal(Int_t index, Double_t value, Bool_t verbose=kFALSE) ;
  void setPdfParamErr(Int_t index, Double_t value) ;
  void setPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal) ;
  void clearPdfParamAsymErr(Int_t index) ;

private:

  Int_t       _printLevel ;
  Int_t       _warnLevel ;
  Int_t       _status ;
  Bool_t      _optConst ;
  Bool_t      _profile ;
  Bool_t      _handleLocalErrors ;
  Int_t       _numBadNLL ;
  Int_t       _nPar ;
  RooArgList* _floatParamList ;
  RooArgList* _initFloatParamList ;
  RooArgList* _constParamList ;
  RooArgList* _initConstParamList ;
  RooAbsReal* _func ;

  Double_t    _maxFCN ;  
  ofstream*   _logfile ;
  Bool_t      _verbose ;
  TStopwatch  _timer ;
  TStopwatch  _cumulTimer ;

  RooMinuit(const RooMinuit&) ;
	
  ClassDef(RooMinuit,0) // RooFit minimizer based on MINUIT
} ;


#endif

