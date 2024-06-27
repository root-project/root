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
#include <memory> // shared_ptr, unique_ptr
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
class RooDataSet;

class RooMinimizer : public TObject {
public:
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

      bool verbose = false;           // local config
      bool profile = false;           // local config
      bool timingAnalysis = false;    // local config
      std::string minimizerType;      // local config
   private:
      int getDefaultWorkers();
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

   int migrad(bool seedingOnly);
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
   /// Enable or disable the logging of function evaluations to a RooDataSet.
   /// \see RooMinimizer::getLogDataSet().
   /// param[in] flag Boolean flag to disable or enable the functionality.
   void setLoggingToDataSet(bool flag = true) { _loggingToDataSet = flag; }

   /// If logging of function evaluations to a RooDataSet is enabled, returns a
   /// pointer to a dataset with one row per evaluation of the RooAbsReal passed
   /// to the minimizer. As columns, there are all floating parameters and the
   /// values they had for that evaluation.
   /// \see RooMinimizer::setLoggingToDataSet(bool).
   RooDataSet *getLogDataSet() const { return _logDataSet.get(); }

   static int getPrintLevel();

   void setMinimizerType(std::string const &type);
   std::string const &minimizerType() const { return _cfg.minimizerType; }

   static void cleanup();
   static RooFit::OwningPtr<RooFitResult> lastMinuitFit();
   static RooFit::OwningPtr<RooFitResult> lastMinuitFit(const RooArgList &varList);

   void saveStatus(const char *label, int status)
   {
      _statusHistory.emplace_back(label, status);
   }

   /// Clears the Minuit status history.
   void clearStatusHistory()
   {
      _statusHistory.clear();
   }

   int evalCounter() const;
   void zeroEvalCount();

   ROOT::Fit::Fitter *fitter();
   const ROOT::Fit::Fitter *fitter() const;

   ROOT::Math::IMultiGenFunction *getMultiGenFcn() const;

   int getNPar() const;

   void applyCovarianceMatrix(TMatrixDSym const &V);

private:
   friend class RooAbsMinimizerFcn;
   friend class RooMinimizerFcn;

   std::unique_ptr<RooAbsReal::EvalErrorContext> makeEvalErrorContext() const;

   void addParamsToProcessTimer();

   void profileStart();
   void profileStop();

   std::ofstream *logfile();
   double &maxFCN();

   bool fitFcn() const;

   // constructor helper functions
   void initMinimizerFirstPart();
   void initMinimizerFcnDependentPart(double defaultErrorLevel);

   int _status = -99;
   bool _profileStart = false;
   bool _loggingToDataSet = false;

   TStopwatch _timer;
   TStopwatch _cumulTimer;

   std::unique_ptr<TMatrixDSym> _extV;

   std::unique_ptr<RooAbsMinimizerFcn> _fcn;

   static std::unique_ptr<ROOT::Fit::Fitter> _theFitter;

   std::vector<std::pair<std::string, int>> _statusHistory;

   std::unique_ptr<RooDataSet> _logDataSet;

   RooMinimizer::Config _cfg; // local config object

   ClassDefOverride(RooMinimizer, 0) // RooFit interface to ROOT::Fit::Fitter
};

#endif
