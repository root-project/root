/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   VC, Vince Croft,     DIANA / NYU,        vincent.croft@cern.ch          *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_GRAD_MINIMIZER
#define ROO_GRAD_MINIMIZER

#include "TObject.h"
#include "TStopwatch.h"
#include <fstream>
#include "TMatrixDSymfwd.h"


#include "Fit/Fitter.h"
//#include "RooMinimizerFcn.h"
//#include "RooGradMinimizerFcn.h"

class RooAbsReal ;
class RooFitResult ;
class RooArgList ;
class RooRealVar ;
class RooArgSet ;
class TH2F ;
class RooPlot ;

class RooGradMinimizerFcn;

class RooGradMinimizer : public TObject {
public:

  RooGradMinimizer(RooAbsReal& function) ;
  virtual ~RooGradMinimizer() ;

  enum Strategy { Speed=0, Balance=1, Robustness=2 } ;
  enum PrintLevel { None=-1, Reduced=0, Normal=1, ExtraForProblem=2, Maximum=3 } ;

  Int_t setPrintLevel(Int_t newLevel);
  void setStrategy(Int_t strat);
  void setErrorLevel(Double_t level);
  void setEps(Double_t eps);
  void optimizeConst(Int_t flag) ;

  RooFitResult* fit(const char* options) ;

  Int_t migrad() ;
  Int_t hesse() ;
  Int_t minos() ;
  Int_t minos(const RooArgSet& minosParamList) ;

  Int_t minimize(const char* type, const char* alg=0) ;

  void setMinimizerType(const char* type);

  static void cleanup() ;
  static RooFitResult* lastMinuitFit(const RooArgList& varList=RooArgList());

  void saveStatus(const char* label, Int_t status) { _statusHistory.push_back(std::pair<std::string,int>(label,status)) ; }

  Int_t evalCounter() const; // { return fitterFcn()->evalCounter() ; }
  void zeroEvalCount(); // { fitterFcn()->zeroEvalCount() ; }

  ROOT::Fit::Fitter* fitter() ;
  const ROOT::Fit::Fitter* fitter() const ;

  void setVerbose(Bool_t flag=kTRUE);

protected:

  friend class RooAbsPdf ;

  void profileStart() ;
  void profileStop() ;

  inline Int_t getNPar() const; // { return fitterFcn()->NDim() ; }
  inline std::ofstream* logfile(); // { return fitterFcn()->GetLogFile(); }
  inline Double_t& maxFCN(); // { return fitterFcn()->GetMaxFCN() ; }
  
  const RooGradMinimizerFcn* fitterFcn() const; // {  return ( fitter()->GetFCN() ? (dynamic_cast<RooGradMinimizerFcn*>(fitter()->GetFCN())) : _fcn ) ; }
  RooGradMinimizerFcn* fitterFcn(); // { return ( fitter()->GetFCN() ? (dynamic_cast<RooGradMinimizerFcn*>(fitter()->GetFCN())) : _fcn ) ; }

private:

  Int_t       _printLevel = 1;
  Int_t       _status ;
  Bool_t      _optConst = kFALSE;
  Bool_t      _profile = kFALSE;
  RooAbsReal* _func ;

  Bool_t      _verbose = kFALSE;
  TStopwatch  _timer ;
  TStopwatch  _cumulTimer ;
  Bool_t      _profileStart = kFALSE;

  TMatrixDSym* _extV = 0;

  RooGradMinimizerFcn *_fcn;
  std::string _minimizerType = "Minuit2";

  static ROOT::Fit::Fitter *_theFitter ;

  std::vector<std::pair<std::string,int> > _statusHistory ;

  RooGradMinimizer(const RooGradMinimizer&) ;
	
} ;

#endif
