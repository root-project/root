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

#include <TStopwatch.h>
#include <TMatrixDSymfwd.h>

#include <Fit/FitConfig.h>

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class RooAbsMinimizerFcn;
class RooAbsReal;
class RooFitResult;
class RooArgList;
class RooRealVar;
class RooArgSet;
class RooPlot;
namespace RooFit {
namespace TestStatistics {
class LikelihoodGradientJob;
}
} // namespace RooFit

class RooMinimizer : public TObject {
public:
   // Internal struct for the temporary fit result.
   struct FitResult {

      FitResult() = default;
      FitResult(const ROOT::Fit::FitConfig &fconfig);

      double error(unsigned int i) const { return (i < fErrors.size()) ? fErrors[i] : 0; }
      double lowerError(unsigned int i) const;
      double upperError(unsigned int i) const;

      double Edm() const { return fEdm; }
      bool IsValid() const { return fValid; }
      int Status() const { return fStatus; }
      void GetCovarianceMatrix(TMatrixDSym &cov) const;

      bool isParameterFixed(unsigned int ipar) const;

      bool fValid = false;                       ///< flag for indicating valid fit
      int fStatus = -1;                          ///< minimizer status code
      int fCovStatus = -1;                       ///< covariance matrix status code
      double fVal = 0;                           ///< minimum function value
      double fEdm = -1;                          ///< expected distance from minimum
      std::map<unsigned int, bool> fFixedParams; ///< list of fixed parameters
      std::vector<double> fParams;               ///< parameter values. Size is total number of parameters
      std::vector<double> fErrors;               ///< errors
      std::vector<double> fCovMatrix; ///< covariance matrix (size is npar*(npar+1)/2) where npar is total parameters
      std::vector<double> fGlobalCC;  ///< global Correlation coefficient
      std::map<unsigned int, std::pair<double, double>> fMinosErrors; ///< map contains the two Minos errors
      std::string fMinimType;                                         ///< string indicating type of minimizer
   };

   /// Config argument to RooMinimizer constructor.
   struct Config {

      Config() {}

      bool useGradient = true; // Use the gradient provided by the RooAbsReal, if there is one.

      double recoverFromNaN = 10.; // RooAbsMinimizerFcn config
      int printEvalErrors = 10;    // RooAbsMinimizerFcn config
      int doEEWall = 1;            // RooAbsMinimizerFcn config
      int offsetting = -1;         // RooAbsMinimizerFcn config
      const char *logf = nullptr;  // RooAbsMinimizerFcn config

      // RooAbsMinimizerFcn config that can only be set in constructor, 0 means no parallelization (default),
      // -1 is parallelization with the number of workers controlled by RooFit::MultiProcess which
      // defaults to the number of available processors, n means parallelization with n CPU's
      int parallelize = 0;

      // Experimental: RooAbsMinimizerFcn config that can only be set in constructor
      // argument is ignored when parallelize is 0
      bool enableParallelGradient = true;

      // Experimental: RooAbsMinimizerFcn config that can only be set in constructor
      // argument is ignored when parallelize is 0
      bool enableParallelDescent = false;

      bool verbose = false;        // local config
      bool profile = false;        // local config
      bool timingAnalysis = false; // local config
      std::string minimizerType;   // local config

      bool setInitialCovariance = false; // Use covariance matrix provided by user
   };

   // For backwards compatibility with when the RooMinimizer used the ROOT::Math::Fitter.
   class FitterInterface {
   public:
      FitterInterface(ROOT::Fit::FitConfig *config, ROOT::Math::Minimizer *minimizer, FitResult const *result)
         : _config{config}, _minimizer{minimizer}, _result{result}
      {
      }

      ROOT::Fit::FitConfig &Config() const { return *_config; }
      ROOT::Math::Minimizer *GetMinimizer() const { return _minimizer; }
      const FitResult &Result() const { return *_result; }

   private:
      ROOT::Fit::FitConfig *_config = nullptr;
      ROOT::Math::Minimizer *_minimizer = nullptr;
      FitResult const *_result = nullptr;
   };

   explicit RooMinimizer(RooAbsReal &function, Config const &cfg = {});

   ~RooMinimizer() override;

   enum Strategy { Speed = 0, Balance = 1, Robustness = 2 };
   enum PrintLevel { None = -1, Reduced = 0, Normal = 1, ExtraForProblem = 2, Maximum = 3 };

   // Setters on _theFitter
   void setStrategy(int istrat);
   void setErrorLevel(double level);
   void setEps(double eps);
   void setMaxIterations(int n);
   void setMaxFunctionCalls(int n);
   void setPrintLevel(int newLevel);

   // Setters on _fcn
   void optimizeConst(int flag);
   void setEvalErrorWall(bool flag) { _cfg.doEEWall = flag; }
   void setRecoverFromNaNStrength(double strength);
   void setOffsetting(bool flag);
   void setPrintEvalErrors(int numEvalErrors) { _cfg.printEvalErrors = numEvalErrors; }
   void setVerbose(bool flag = true) { _cfg.verbose = flag; }
   bool setLogFile(const char *logf = nullptr);

   int migrad();
   int hesse();
   int minos();
   int minos(const RooArgSet &minosParamList);
   int seek();
   int simplex();
   int improve();

   int minimize(const char *type, const char *alg = nullptr);

   RooFit::OwningPtr<RooFitResult> save(const char *name = nullptr, const char *title = nullptr);
   RooPlot *contour(RooRealVar &var1, RooRealVar &var2, double n1 = 1.0, double n2 = 2.0, double n3 = 0.0,
                    double n4 = 0.0, double n5 = 0.0, double n6 = 0.0, unsigned int npoints = 50);

   void setProfile(bool flag = true) { _cfg.profile = flag; }

   int getPrintLevel();

   void setMinimizerType(std::string const &type);
   std::string const &minimizerType() const { return _cfg.minimizerType; }

   RooFit::OwningPtr<RooFitResult> lastMinuitFit();

   void saveStatus(const char *label, int status) { _statusHistory.emplace_back(label, status); }

   /// Clears the Minuit status history.
   void clearStatusHistory() { _statusHistory.clear(); }

   int evalCounter() const;
   void zeroEvalCount();

   /// Return underlying ROOT fitter object
   inline auto fitter() { return std::make_unique<FitterInterface>(&_config, _minimizer.get(), _result.get()); }

   ROOT::Math::IMultiGenFunction *getMultiGenFcn() const;

   int getNPar() const;

   void applyCovarianceMatrix(TMatrixDSym const &V);

private:
   friend class RooAbsMinimizerFcn;
   friend class RooMinimizerFcn;
   friend class RooFit::TestStatistics::LikelihoodGradientJob;

   std::unique_ptr<RooAbsReal::EvalErrorContext> makeEvalErrorContext() const;

   void addParamsToProcessTimer();

   void profileStart();
   void profileStop();

   std::ofstream *logfile();
   double &maxFCN();
   double &fcnOffset() const;

   // constructor helper functions
   void initMinimizerFirstPart();
   void initMinimizerFcnDependentPart(double defaultErrorLevel);

   void determineStatus(bool fitterReturnValue);

   int exec(std::string const &algoName, std::string const &statusName);

   bool fitFCN(const ROOT::Math::IMultiGenFunction &fcn);

   bool calculateHessErrors();
   bool calculateMinosErrors();

   void initMinimizer();
   void updateFitConfig();
   bool updateMinimizerOptions(bool canDifferentMinim = true);

   void fillResult(bool isValid);
   bool update(bool isValid);

   void fillCorrMatrix(RooFitResult &fitRes);
   void updateErrors();

   ROOT::Fit::FitConfig _config;                      ///< fitter configuration (options and parameter settings)
   std::unique_ptr<FitResult> _result;                ///<! pointer to the object containing the result of the fit
   std::unique_ptr<ROOT::Math::Minimizer> _minimizer; ///<! pointer to used minimizer
   int _status = -99;
   bool _profileStart = false;
   TStopwatch _timer;
   TStopwatch _cumulTimer;
   std::unique_ptr<TMatrixDSym> _extV;
   std::unique_ptr<RooAbsMinimizerFcn> _fcn;
   std::vector<std::pair<std::string, int>> _statusHistory;
   RooMinimizer::Config _cfg; // local config object

   ClassDefOverride(RooMinimizer, 0) // RooFit interface to ROOT::Math::Minimizer
};

#endif
