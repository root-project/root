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

#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooFit/TestStatistics/LikelihoodWrapper.h>
#include <RooFit/TestStatistics/LikelihoodGradientWrapper.h>

#include <Fit/Fitter.h>
#include <TStopwatch.h>
#include <TMatrixDSymfwd.h>

#include <fstream>
#include <memory>  // shared_ptr, unique_ptr
#include <string>
#include <utility>
#include <vector>

class RooAbsMinimizerFcn ;
class RooAbsReal ;
class RooFitResult ;
class RooArgList ;
class RooRealVar ;
class RooArgSet ;
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
  void setEvalErrorWall(bool flag) ;
  void setRecoverFromNaNStrength(double strength) ;
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
  void setPrintEvalErrors(int numEvalErrors) ;
  void setVerbose(bool flag=true) ;
  void setProfile(bool flag=true) { _profile = flag ; }
  bool setLogFile(const char* logf=nullptr) ;

  int getPrintLevel() const { return _printLevel; }

  void setMinimizerType(const char* type) ;

  static void cleanup() ;
  static RooFitResult* lastMinuitFit() ;
  static RooFitResult* lastMinuitFit(const RooArgList& varList) ;

  void saveStatus(const char* label, int status) { _statusHistory.push_back(std::pair<std::string,int>(label,status)) ; }

  int evalCounter() const {return _evalCounter; }
  void zeroEvalCount() { _evalCounter = 0; }

  ROOT::Fit::Fitter* fitter() ;
  const ROOT::Fit::Fitter* fitter() const ;

  ROOT::Math::IMultiGenFunction* getMultiGenFcn() const;

  int getNPar() const ;

  void applyCovarianceMatrix(TMatrixDSym const& V) ;

private:

  friend class RooAbsMinimizerFcn;

  void incrementEvalCounter() { _evalCounter++; }

  void profileStart() ;
  void profileStop() ;

  std::ofstream* logfile() ;
  double& maxFCN() ;

  bool fitFcn() const;

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
  int _evalCounter = 0;

  ClassDefOverride(RooMinimizer,0) // RooFit interface to ROOT::Fit::Fitter
} ;

#endif
