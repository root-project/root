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

#include "RooArgList.h" // cannot just use forward decl due to default argument in lastMinuitFit

#include "RooMinimizerFcn.h"
#include "RooGradMinimizerFcn.h"
#include "TestStatistics/RooAbsL.h"

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
namespace RooFit {
namespace TestStatistics {
class MinuitFcnGrad;  // this one is necessary due to circular include dependencies
class LikelihoodSerial;
class LikelihoodGradientSerial;
}
} // namespace RooFit

class RooMinimizer : public TObject {
public:
  enum class FcnMode { classic, gradient, generic_wrapper };

  explicit RooMinimizer(RooAbsReal &function, FcnMode fcnMode = FcnMode::classic);
  static std::unique_ptr<RooMinimizer> create(RooAbsReal &function, FcnMode fcnMode = FcnMode::classic);
  template <typename LikelihoodWrapperT = RooFit::TestStatistics::LikelihoodSerial,
            typename LikelihoodGradientWrapperT = RooFit::TestStatistics::LikelihoodGradientSerial>
  static std::unique_ptr<RooMinimizer> create(std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood);

  ~RooMinimizer() override;

  enum Strategy { Speed=0, Balance=1, Robustness=2 } ;
  enum PrintLevel { None=-1, Reduced=0, Normal=1, ExtraForProblem=2, Maximum=3 } ;
  void setStrategy(Int_t strat) ;
  void setErrorLevel(Double_t level) ;
  void setEps(Double_t eps) ;
  void optimizeConst(Int_t flag) ;
  void setEvalErrorWall(Bool_t flag) { fitterFcn()->SetEvalErrorWall(flag); }
  /// \copydoc RooMinimizerFcn::SetRecoverFromNaNStrength()
  void setRecoverFromNaNStrength(double strength) { fitterFcn()->SetRecoverFromNaNStrength(strength); }
  void setOffsetting(Bool_t flag) ;
  void setMaxIterations(Int_t n) ;
  void setMaxFunctionCalls(Int_t n) ;

  RooFitResult* fit(const char* options) ;

  Int_t migrad() ;
  Int_t hesse() ;
  Int_t minos() ;
  Int_t minos(const RooArgSet& minosParamList) ;
  Int_t seek() ;
  Int_t simplex() ;
  Int_t improve() ;

  Int_t minimize(const char* type, const char* alg=0) ;

  RooFitResult* save(const char* name=0, const char* title=0) ;
  RooPlot* contour(RooRealVar& var1, RooRealVar& var2, 
		   Double_t n1=1, Double_t n2=2, Double_t n3=0,
		   Double_t n4=0, Double_t n5=0, Double_t n6=0, unsigned int npoints = 50) ;

  Int_t setPrintLevel(Int_t newLevel) ; 
  void setPrintEvalErrors(Int_t numEvalErrors) { fitterFcn()->SetPrintEvalErrors(numEvalErrors); }
  void setVerbose(Bool_t flag=kTRUE) { _verbose = flag ; fitterFcn()->SetVerbose(flag); }
  void setProfile(Bool_t flag=kTRUE) { _profile = flag ; }
  Bool_t setLogFile(const char* logf=0) { return fitterFcn()->SetLogFile(logf); }

  Int_t getPrintLevel() const;

  void setMinimizerType(const char* type) ;

  static void cleanup() ;
  static RooFitResult* lastMinuitFit(const RooArgList& varList=RooArgList()) ;

  void saveStatus(const char* label, Int_t status) { _statusHistory.push_back(std::pair<std::string,int>(label,status)) ; }

  Int_t evalCounter() const { return fitterFcn()->evalCounter() ; }
  void zeroEvalCount() { fitterFcn()->zeroEvalCount() ; }

  ROOT::Fit::Fitter* fitter() ;
  const ROOT::Fit::Fitter* fitter() const ;

  ROOT::Math::IMultiGenFunction* getFitterMultiGenFcn() const;
  ROOT::Math::IMultiGenFunction* getMultiGenFcn() const;

protected:

  friend class RooAbsPdf ;
  void applyCovarianceMatrix(TMatrixDSym& V) ;

  void profileStart() ;
  void profileStop() ;

  inline Int_t getNPar() const { return fitterFcn()->getNDim() ; }
  inline std::ofstream* logfile() { return fitterFcn()->GetLogFile(); }
  inline Double_t& maxFCN() { return fitterFcn()->GetMaxFCN() ; }

  const RooAbsMinimizerFcn *fitterFcn() const;
  RooAbsMinimizerFcn *fitterFcn();

  bool fitFcn() const;

private:
  template <typename LikelihoodWrapperT = RooFit::TestStatistics::LikelihoodSerial, typename LikelihoodGradientWrapperT = RooFit::TestStatistics::LikelihoodGradientSerial>
  RooMinimizer(std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood,
               LikelihoodWrapperT* /* used only for template deduction */ = static_cast<RooFit::TestStatistics::LikelihoodSerial*>(nullptr),
               LikelihoodGradientWrapperT* /* used only for template deduction */ = static_cast<RooFit::TestStatistics::LikelihoodGradientSerial*>(nullptr));

  Int_t _printLevel = 1;
  Int_t _status = -99;
  Bool_t _profile = kFALSE;

  Bool_t _verbose = kFALSE;
  TStopwatch _timer;
  TStopwatch _cumulTimer;
  Bool_t _profileStart = kFALSE;

  TMatrixDSym *_extV = 0;

  RooAbsMinimizerFcn *_fcn;
  std::string _minimizerType = "Minuit";
  FcnMode _fcnMode;

  static ROOT::Fit::Fitter *_theFitter ;

  std::vector<std::pair<std::string,int> > _statusHistory ;

  RooMinimizer(const RooMinimizer&) ;
	
  ClassDefOverride(RooMinimizer,0) // RooFit interface to ROOT::Fit::Fitter
} ;

// include here to avoid circular dependency issues in class definitions
#include "TestStatistics/MinuitFcnGrad.h"

template <typename LikelihoodWrapperT, typename LikelihoodGradientWrapperT>
RooMinimizer::RooMinimizer(std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood, LikelihoodWrapperT* /* value unused */,
                           LikelihoodGradientWrapperT* /* value unused */) :
   _fcnMode(FcnMode::generic_wrapper)
{
   RooSentinel::activate();

   if (_theFitter)
      delete _theFitter;
   _theFitter = new ROOT::Fit::Fitter;
   _theFitter->Config().SetMinimizer(_minimizerType.c_str());
   setEps(1.0); // default tolerance

   _fcn = RooFit::TestStatistics::MinuitFcnGrad::create<LikelihoodWrapperT, LikelihoodGradientWrapperT>(likelihood, this, _verbose);

   // default max number of calls
   _theFitter->Config().MinimizerOptions().SetMaxIterations(500 * _fcn->getNDim());
   _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(500 * _fcn->getNDim());

   // Shut up for now
   setPrintLevel(-1);

   // Use +0.5 for 1-sigma errors
   setErrorLevel(likelihood->defaultErrorLevel());

   // Declare our parameters to MINUIT
   _fcn->Synchronize(_theFitter->Config().ParamsSettings(), _fcn->getOptConst(), _verbose);

   // Now set default verbosity
   if (RooMsgService::instance().silentMode()) {
      setPrintLevel(-1);
   } else {
      setPrintLevel(1);
   }
}

// static function
template <typename LikelihoodWrapperT, typename LikelihoodGradientWrapperT>
std::unique_ptr<RooMinimizer> RooMinimizer::create(std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood) {
   return std::unique_ptr<RooMinimizer>(new RooMinimizer(likelihood, static_cast<LikelihoodWrapperT*>(nullptr),
                                                         static_cast<LikelihoodGradientWrapperT*>(nullptr)));
}

#endif
