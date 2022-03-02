/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_MINIMIZER
#define ROO_MINIMIZER

#include <memory>  // shared_ptr, unique_ptr

#include "TObject.h"
#include "TStopwatch.h"
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include "TMatrixDSymfwd.h"

#include <RooAbsMinimizerFcn.h>
#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooFit/TestStatistics/LikelihoodWrapper.h>
#include <RooFit/TestStatistics/LikelihoodGradientWrapper.h>

#include "RooSentinel.h"
#include "RooMsgService.h"

#include "Fit/Fitter.h"
#include <stdexcept> // logic_error

class RooAbsReal ;
class RooFitResult ;
class RooArgList ;
class RooRealVar ;
class RooArgSet ;
class TH2F ;
class RooPlot ;

class RooMinimizer : public TObject {
public:
  enum class FcnMode { classic, gradient, generic_wrapper };

  explicit RooMinimizer(RooAbsReal &function, FcnMode fcnMode = FcnMode::classic);
  explicit RooMinimizer(std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood,
                        RooFit::TestStatistics::LikelihoodMode likelihoodMode =
                           RooFit::TestStatistics::LikelihoodMode::serial,
                        RooFit::TestStatistics::LikelihoodGradientMode likelihoodGradientMode =
                           RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);

  ~RooMinimizer() override;

  enum Strategy { Speed=0, Balance=1, Robustness=2 } ;
  enum PrintLevel { None=-1, Reduced=0, Normal=1, ExtraForProblem=2, Maximum=3 } ;
  void setStrategy(int strat) ;
  void setErrorLevel(double level) ;
  void setEps(double eps) ;
  void optimizeConst(int flag) ;
  void setEvalErrorWall(bool flag) { fitterFcn()->SetEvalErrorWall(flag); }
  /// \copydoc RooMinimizerFcn::SetRecoverFromNaNStrength()
  void setRecoverFromNaNStrength(double strength) { fitterFcn()->SetRecoverFromNaNStrength(strength); }
  void setOffsetting(bool flag) ;
  void setMaxIterations(int n) ;
  void setMaxFunctionCalls(int n) ;

  int migrad() ;
  int hesse() ;
  int minos() ;
  int minos(const RooArgSet& minosParamList) ;
  int seek() ;
  int simplex() ;
  int improve() ;

  int minimize(const char* type, const char* alg=0) ;

  RooFitResult* save(const char* name=0, const char* title=0) ;
  RooPlot* contour(RooRealVar& var1, RooRealVar& var2,
         double n1=1, double n2=2, double n3=0,
         double n4=0, double n5=0, double n6=0, unsigned int npoints = 50) ;

  int setPrintLevel(int newLevel) ;
  void setPrintEvalErrors(int numEvalErrors) { fitterFcn()->SetPrintEvalErrors(numEvalErrors); }
  void setVerbose(bool flag=true) { _verbose = flag ; fitterFcn()->SetVerbose(flag); }
  void setProfile(bool flag=true) { _profile = flag ; }
  bool setLogFile(const char* logf=nullptr) { return fitterFcn()->SetLogFile(logf); }

  int getPrintLevel() const { return _printLevel; }

  void setMinimizerType(const char* type) ;

  static void cleanup() ;
  static RooFitResult* lastMinuitFit() ;
  static RooFitResult* lastMinuitFit(const RooArgList& varList) ;

  void saveStatus(const char* label, int status) { _statusHistory.push_back(std::pair<std::string,int>(label,status)) ; }

  int evalCounter() const { return fitterFcn()->evalCounter() ; }
  void zeroEvalCount() { fitterFcn()->zeroEvalCount() ; }

  ROOT::Fit::Fitter* fitter() ;
  const ROOT::Fit::Fitter* fitter() const ;

  ROOT::Math::IMultiGenFunction* getFitterMultiGenFcn() const;
  ROOT::Math::IMultiGenFunction* getMultiGenFcn() const;

  inline int getNPar() const { return fitterFcn()->getNDim() ; }

protected:

  friend class RooAbsPdf ;
  void applyCovarianceMatrix(TMatrixDSym& V) ;

  void profileStart() ;
  void profileStop() ;

  inline std::ofstream* logfile() { return fitterFcn()->GetLogFile(); }
  inline double& maxFCN() { return fitterFcn()->GetMaxFCN() ; }

  const RooAbsMinimizerFcn *fitterFcn() const;
  RooAbsMinimizerFcn *fitterFcn();

  bool fitFcn() const;

private:
  // constructor helper functions
  void initMinimizerFirstPart();
  void initMinimizerFcnDependentPart(double defaultErrorLevel);

  int _printLevel = 1;
  int _status = -99;
  bool _profile = false;

  bool _verbose = false;
  TStopwatch _timer;
  TStopwatch _cumulTimer;
  bool _profileStart = false;

  std::unique_ptr<TMatrixDSym> _extV;

  RooAbsMinimizerFcn *_fcn;
  std::string _minimizerType = "Minuit";
  FcnMode _fcnMode;

  static std::unique_ptr<ROOT::Fit::Fitter> _theFitter ;

  std::vector<std::pair<std::string,int> > _statusHistory ;

  RooMinimizer(const RooMinimizer&) ;

  ClassDefOverride(RooMinimizer,0) // RooFit interface to ROOT::Fit::Fitter
} ;

#endif
