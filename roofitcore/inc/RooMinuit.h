/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_MINUIT
#define ROO_MINUIT

#include "TObject.h"
#include "TStopwatch.h"

class RooAbsReal ;
class RooFitResult ;
class ofstream ;
class RooArgList ;
class RooRealVar ;

class RooMinuit : public TObject {
public:

  RooMinuit(RooAbsReal& function) ;
  virtual ~RooMinuit() ;

  void setStrategy(Int_t strat) ;
  void setErrorLevel(Double_t level) ;
  void optimizeConst(Bool_t flag) ;

  Int_t fit(const char* options) ;

  Int_t migrad() ;
  Int_t hesse() ;
  Int_t minos() ;
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

private:

  Int_t       _printLevel ;
  Int_t       _warnLevel ;
  Int_t       _status ;
  Bool_t      _optConst ;
  Bool_t      _profile ;
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

