/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMinuit.h,v 1.15 2007/07/12 20:30:28 wouter Exp $
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
#include "TMatrixDSymfwd.h"
#include <vector>
#include <string>
#include <utility>

#include <ROOT/RConfig.hxx>

class RooAbsReal ;
class RooFitResult ;
class RooArgList ;
class RooRealVar ;
class RooArgSet ;
class RooAbsArg ;
class TVirtualFitter ;
class TH2F ;
class RooPlot ;

void RooMinuitGlue(Int_t& /*np*/, double* /*gin*/,  double &f, double *par, Int_t /*flag*/) ;

class RooMinuit : public TObject {
public:

  RooMinuit(RooAbsReal& function) ;
  ~RooMinuit() override ;

  enum Strategy { Speed=0, Balance=1, Robustness=2 } ;
  enum PrintLevel { None=-1, Reduced=0, Normal=1, ExtraForProblem=2, Maximum=3 } ;
  void setStrategy(Int_t strat) ;
  void setErrorLevel(double level) ;
  void setEps(double eps) ;
  void optimizeConst(Int_t flag) ;
  void setEvalErrorWall(bool flag) { _doEvalErrorWall = flag ; }
  void setOffsetting(bool flag) ;

  RooFitResult* fit(const char* options) ;

  Int_t migrad() ;
  Int_t hesse() ;
  Int_t minos() ;
  Int_t minos(const RooArgSet& minosParamList) ;  // added FMV, 08/18/03
  Int_t seek() ;
  Int_t simplex() ;
  Int_t improve() ;

  RooFitResult* save(const char* name=nullptr, const char* title=nullptr) ;
  RooPlot* contour(RooRealVar& var1, RooRealVar& var2,
         double n1=1, double n2=2, double n3=0.0,
         double n4=0.0, double n5=0.0, double n6=0.0) ;

  Int_t setPrintLevel(Int_t newLevel) ;
  void setNoWarn() ;
  Int_t setWarnLevel(Int_t newLevel) ;
  void setPrintEvalErrors(Int_t numEvalErrors) { _printEvalErrors = numEvalErrors ; }
  void setVerbose(bool flag=true) { _verbose = flag ; }
  void setProfile(bool flag=true) { _profile = flag ; }
  void setMaxEvalMultiplier(Int_t n) { _maxEvalMult = n ; }
  bool setLogFile(const char* logfile=nullptr) ;

  static void cleanup() ;

  Int_t evalCounter() const { return _evalCounter ; }
  void zeroEvalCount() { _evalCounter = 0 ; }

protected:

  friend class RooAbsPdf ;
  void applyCovarianceMatrix(TMatrixDSym& V) ;

  friend void RooMinuitGlue(Int_t &np, double *gin, double &f, double *par, Int_t flag) ;

  void profileStart() ;
  void profileStop() ;

  bool synchronize(bool verbose) ;
  void backProp() ;

  inline Int_t getNPar() const { return _nPar ; }
  inline std::ofstream* logfile() const { return _logfile ; }
  inline double& maxFCN() { return _maxFCN ; }

  double getPdfParamVal(Int_t index) ;
  double getPdfParamErr(Int_t index) ;
  virtual bool setPdfParamVal(Int_t index, double value, bool verbose=false) ;
  void setPdfParamErr(Int_t index, double value) ;
  void setPdfParamErr(Int_t index, double loVal, double hiVal) ;
  void clearPdfParamAsymErr(Int_t index) ;

  void saveStatus(const char* label, Int_t status) { _statusHistory.push_back(std::pair<std::string,int>(label,status)) ; }

  void updateFloatVec() ;

private:

  Int_t       _evalCounter ;
  Int_t       _printLevel ;
  Int_t       _warnLevel ;
  Int_t       _status ;
  Int_t       _optConst ;
  bool      _profile ;
  bool      _handleLocalErrors ;
  Int_t       _numBadNLL ;
  Int_t       _nPar ;
  Int_t       _printEvalErrors ;
  bool      _doEvalErrorWall ;
  Int_t       _maxEvalMult ;
  RooArgList* _floatParamList ;
  std::vector<RooAbsArg*> _floatParamVec ;
  RooArgList* _initFloatParamList ;
  RooArgList* _constParamList ;
  RooArgList* _initConstParamList ;
  RooAbsReal* _func ;

  double    _maxFCN ;
  std::ofstream*   _logfile ;
  bool      _verbose ;
  TStopwatch  _timer ;
  TStopwatch  _cumulTimer ;

  TMatrixDSym* _extV ;

  static TVirtualFitter *_theFitter ;

  std::vector<std::pair<std::string,int> > _statusHistory ;

  RooMinuit(const RooMinuit&) ;

  ClassDefOverride(RooMinuit,0) // RooFit minimizer based on MINUIT
} R__SUGGEST_ALTERNATIVE("Please use RooMinimizer instead of RooMinuit");


#endif

