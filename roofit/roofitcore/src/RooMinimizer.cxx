/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooMinimizer.cxx
\class RooMinimizer
\ingroup Roofitcore

RooMinimizer provides a seamless interface between the minimizer functionality
and the native RooFit interface.
By default the Minimizer is Minuit 2, which RooMinimizer drives directly via the
Minuit2 library interface. Other minimizer types are reached via the
ROOT::Math::Minimizer plugin interface.
RooMinimizer can minimize any RooAbsReal function with respect to
its parameters. Usual choices for minimization are the object returned by
RooAbsPdf::createNLL() or RooAbsReal::createChi2().
RooMinimizer has methods corresponding to MINUIT functions like
hesse(), migrad(), minos() etc. In each of these function calls
the state of the MINUIT engine is synchronized with the state
of the RooFit variables: any change in variables, change
in the constant status etc is forwarded to MINUIT prior to
execution of the MINUIT call. Afterwards the RooFit objects
are resynchronized with the output state of MINUIT: changes
parameter values, errors are propagated.
Various methods are available to control verbosity or profiling.
**/

#include "RooMinimizer.h"

#include "RooAbsMinimizerFcn.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooArgSet.h"
#include "RooCategory.h"
#include "RooDataSet.h"
#include "RooEvaluatorWrapper.h"
#include "RooFit/TestStatistics/RooAbsL.h"
#include "RooFit/TestStatistics/RooRealL.h"
#include "RooFitResult.h"
#include "RooHelpers.h"
#include "RooMinimizerFcn.h"
#include "RooMsgService.h"
#include "RooMultiPdf.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooSentinel.h"
#ifdef ROOFIT_MULTIPROCESS
#include "TestStatistics/MinuitFcnGrad.h"
#include "RooFit/MultiProcess/Config.h"
#include "RooFit/MultiProcess/ProcessTimer.h"
#endif

#include "RooFitImplHelpers.h"

#include <Fit/BasicFCN.h>
#include <Math/IOptions.h>
#include <Math/Minimizer.h>
#include <Minuit2/CombinedMinimizer.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MinosError.h>
#include <Minuit2/MnContours.h>
#include <Minuit2/MnHesse.h>
#include <Minuit2/MnMinos.h>
#include <Minuit2/MnPrint.h>
#include <Minuit2/MnStrategy.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/ScanMinimizer.h>
#include <Minuit2/SimplexMinimizer.h>
#include <Minuit2/VariableMetricMinimizer.h>
#include <TClass.h>
#include <TGraph.h>
#include <TMarker.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept> // logic_error

namespace {

class FreezeDisconnectedParametersRAII {
public:
   FreezeDisconnectedParametersRAII(RooMinimizer const *minimizer, RooAbsMinimizerFcn const &fcn)
      : _minimizer{minimizer}, _frozen{fcn.freezeDisconnectedParameters()}
   {
      if (!_frozen.empty()) {
         oocoutI(_minimizer, Minimization) << "Freezing disconnected parameters: " << _frozen << std::endl;
      }
   }
   ~FreezeDisconnectedParametersRAII()
   {
      if (!_frozen.empty()) {
         oocoutI(_minimizer, Minimization) << "Unfreezing disconnected parameters: " << _frozen << std::endl;
      }
      RooHelpers::setAllConstant(_frozen, false);
   }

private:
   RooMinimizer const *_minimizer = nullptr;
   RooArgSet _frozen;
};

std::vector<std::vector<int>> generateOrthogonalCombinations(const std::vector<int> &maxValues)
{
   std::vector<std::vector<int>> combos;
   std::vector<int> base(maxValues.size(), 0);
   combos.push_back(base);
   for (size_t i = 0; i < maxValues.size(); ++i) {
      for (int v = 1; v < maxValues[i]; ++v) {
         std::vector<int> tmp = base;
         tmp[i] = v;
         combos.push_back(tmp);
      }
   }
   return combos;
}

void reorderCombinations(std::vector<std::vector<int>> &combos, const std::vector<int> &max,
                         const std::vector<int> &base)
{
   for (auto &combo : combos) {
      for (size_t i = 0; i < combo.size(); ++i) {
         combo[i] = (combo[i] + base[i]) % max[i];
      }
   }
}

// The RooEvaluatorWrapper uses its own logic to decide what needs to be
// re-evaluated. We can therefore disable the regular dirty state propagation
// temporarily during minimization. However, some RooAbsArgs shared with other
// regular RooFit computation graphs outside the minimized likelihood, so we
// have to make sure that the operation mode is reset after the minimization.
//
// This should be called before running any routine via the _minimizer data
// member. The RAII object should only be destructed after the routine is done.
std::unique_ptr<ChangeOperModeRAII> setOperModesDirty(RooAbsReal &function)
{
   if (auto *wrapper = dynamic_cast<RooFit::Experimental::RooEvaluatorWrapper *>(&function)) {
      return wrapper->setOperModes(RooAbsArg::ADirty);
   }
   return {};
}

// Set the global Minuit2 print level for the duration of a Minuit2 call.
class Minuit2PrintLevelRAII {
public:
   Minuit2PrintLevelRAII(int level) : _prevLevel{ROOT::Minuit2::MnPrint::SetGlobalLevel(level)} {}
   ~Minuit2PrintLevelRAII() { ROOT::Minuit2::MnPrint::SetGlobalLevel(_prevLevel); }

private:
   int _prevLevel;
};

// The Minuit2-specific extra options, either set on the minimizer options
// explicitly or taken from the global defaults for Minuit2.
ROOT::Math::IOptions const *minuit2ExtraOptions(ROOT::Math::MinimizerOptions const &options)
{
   if (auto *opts = options.ExtraOptions()) {
      return opts;
   }
   return ROOT::Math::MinimizerOptions::FindDefault("Minuit2");
}

// Build the Minuit2 strategy from the strategy level and the Minuit2-specific
// extra options.
ROOT::Minuit2::MnStrategy makeMinuit2Strategy(ROOT::Math::MinimizerOptions const &options)
{
   ROOT::Minuit2::MnStrategy st{static_cast<unsigned int>(options.Strategy())};
   ROOT::Math::IOptions const *minuit2Opt = minuit2ExtraOptions(options);
   if (!minuit2Opt) {
      return st;
   }
   auto customize = [&minuit2Opt](const char *name, auto val) {
      minuit2Opt->GetValue(name, val);
      return val;
   };
   st.SetGradientNCycles(customize("GradientNCycles", int(st.GradientNCycles())));
   st.SetHessianNCycles(customize("HessianNCycles", int(st.HessianNCycles())));
   st.SetHessianGradientNCycles(customize("HessianGradientNCycles", int(st.HessianGradientNCycles())));

   st.SetGradientTolerance(customize("GradientTolerance", st.GradientTolerance()));
   st.SetGradientStepTolerance(customize("GradientStepTolerance", st.GradientStepTolerance()));
   st.SetHessianStepTolerance(customize("HessianStepTolerance", st.HessianStepTolerance()));
   st.SetHessianG2Tolerance(customize("HessianG2Tolerance", st.HessianG2Tolerance()));

   st.SetHessianCentralFDMixedDerivatives(
      customize("HessianCentralFDMixedDerivatives", int(st.HessianCentralFDMixedDerivatives())));
   st.SetHessianForcePosDef(customize("HessianForcePosDef", int(st.HessianForcePosDef())));

   return st;
}

// Create the Minuit2 minimization algorithm for the given algorithm name.
// Returns nullptr for algorithms that RooFit can't use.
std::unique_ptr<ROOT::Minuit2::ModularFunctionMinimizer> makeMinuit2Minimizer(std::string algo)
{
   std::transform(algo.begin(), algo.end(), algo.begin(), [](unsigned char c) { return std::tolower(c); });
   using namespace ROOT::Minuit2;
   if (algo == "simplex")
      return std::make_unique<SimplexMinimizer>();
   if (algo == "minimize")
      return std::make_unique<CombinedMinimizer>();
   if (algo == "scan")
      return std::make_unique<ScanMinimizer>();
   if (algo == "bfgs")
      return std::make_unique<VariableMetricMinimizer>(VariableMetricMinimizer::BFGSType());
   if (algo == "fumili" || algo == "fumili2") {
      // Fumili needs a FumiliFCNBase, which RooFit does not provide.
      return nullptr;
   }
   // MIGRAD is the default algorithm, also for unknown algorithm names.
   return std::make_unique<VariableMetricMinimizer>();
}

bool isFixedOrConst(ROOT::Minuit2::MinuitParameter const &par)
{
   return par.IsFixed() || par.IsConst();
}

// Covariance matrix element in the external parameter space, zero for fixed
// parameters and when no covariance matrix is available.
double minuit2Covariance(ROOT::Minuit2::MnUserParameterState const &state, unsigned int i, unsigned int j)
{
   if (!state.HasCovariance() || isFixedOrConst(state.Parameter(i)) || isFixedOrConst(state.Parameter(j))) {
      return 0.;
   }
   return state.Covariance()(state.IntOfExt(i), state.IntOfExt(j));
}

// Covariance matrix status code with the convention of ROOT::Math::Minimizer:
//  -1: not available (inversion failed or Hesse failed)
//   0: available but not positive defined
//   1: covariance only approximate
//   2: full matrix but forced positive definite
//   3: full accurate matrix
int minuit2CovMatrixStatus(ROOT::Minuit2::FunctionMinimum const *minimum,
                           ROOT::Minuit2::MnUserParameterState const &state)
{
   if (!minimum) {
      return state.CovarianceStatus();
   }
   if (minimum->HasAccurateCovar())
      return 3;
   if (minimum->HasMadePosDefCovar())
      return 2;
   if (minimum->HasValidCovariance())
      return 1;
   if (minimum->HasCovariance())
      return 0;
   return -1;
}

} // namespace

/// State of the direct Minuit2 interface.
struct RooMinimizer::Minuit2State {
   /// Minuit2 parameter state: the starting point for the next minimization,
   /// and the result of the last operation.
   ROOT::Minuit2::MnUserParameterState state;
   /// Result of the last minimization, needed for Hesse, Minos and contours.
   std::unique_ptr<ROOT::Minuit2::FunctionMinimum> minimum;
   /// Status code with the same convention as ROOT::Minuit2::Minuit2Minimizer.
   int status = 0;
   /// Status of the last Minos run.
   int minosStatus = -1;
   /// Whether the parameter state was initialized from the parameter settings.
   bool initialized = false;
};

////////////////////////////////////////////////////////////////////////////////
/// Construct MINUIT interface to given function. Function can be anything,
/// but is typically a -log(likelihood) implemented by RooNLLVar or a chi^2
/// (implemented by RooChi2Var). Other frequent use cases are a RooAddition
/// of a RooNLLVar plus a penalty or constraint term. This class propagates
/// all RooFit information (floating parameters, their values and errors)
/// to MINUIT before each MINUIT call and propagates all MINUIT information
/// back to the RooFit object at the end of each call (updated parameter
/// values, their (asymmetric errors) etc. The default MINUIT error level
/// for HESSE and MINOS error analysis is taken from the defaultErrorLevel()
/// value of the input function.

/// Constructor that accepts all configuration in struct with RooAbsReal likelihood
RooMinimizer::RooMinimizer(RooAbsReal &function, Config const &cfg) : _function{function}, _cfg(cfg)
{
   initMinimizerFirstPart();
   auto nll_real = dynamic_cast<RooFit::TestStatistics::RooRealL *>(&function);
   if (nll_real != nullptr) {
      if (_cfg.parallelize != 0) { // new test statistic with multiprocessing library with
                                   // parallel likelihood or parallel gradient
#ifdef ROOFIT_MULTIPROCESS
         if (!_cfg.enableParallelGradient) {
            // Note that this is necessary because there is currently no serial-mode LikelihoodGradientWrapper.
            // We intend to repurpose RooGradMinimizerFcn to build such a LikelihoodGradientSerial class.
            coutI(InputArguments) << "Modular likelihood detected and likelihood parallelization requested, "
                                  << "also setting parallel gradient calculation mode." << std::endl;
            _cfg.enableParallelGradient = true;
         }
         // If _cfg.parallelize is larger than zero set the number of workers to that value. Otherwise do not do
         // anything and let RooFit::MultiProcess handle the number of workers
         if (_cfg.parallelize > 0)
            RooFit::MultiProcess::Config::setDefaultNWorkers(_cfg.parallelize);
         RooFit::MultiProcess::Config::setTimingAnalysis(_cfg.timingAnalysis);

         _fcn = std::make_unique<RooFit::TestStatistics::MinuitFcnGrad>(
            nll_real->getRooAbsL(), this, _config.ParamsSettings(),
            RooFit::TestStatistics::LikelihoodMode{
               static_cast<RooFit::TestStatistics::LikelihoodMode>(int(_cfg.enableParallelDescent))},
            RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);
#else
         throw std::logic_error(
            "Parallel minimization requested, but multiprocessing is not supported on this platform");
#endif
      } else { // modular test statistic non parallel
         coutW(InputArguments)
            << "Requested modular likelihood without gradient parallelization, some features such as offsetting "
            << "may not work yet. Non-modular likelihoods are more reliable without parallelization." << std::endl;
         // The RooRealL that is used in the case where the modular likelihood is being passed to a RooMinimizerFcn does
         // not have offsetting implemented. Therefore, offsetting will not work in this case. Other features might also
         // not work since the RooRealL was not intended for minimization. Further development is required to make the
         // MinuitFcnGrad also handle serial gradient minimization. The MinuitFcnGrad accepts a RooAbsL and has
         // offsetting implemented, thus omitting the need for RooRealL minimization altogether.
         _fcn = std::make_unique<RooMinimizerFcn>(&function, this);
      }
   } else {
      if (_cfg.parallelize != 0) { // Old test statistic with parallel likelihood or gradient
         throw std::logic_error("In RooMinimizer constructor: Selected likelihood evaluation but a "
                                "non-modular likelihood was given. Please supply ModularL(true) as an "
                                "argument to createNLL for modular likelihoods to use likelihood "
                                "or gradient parallelization.");
      }
      _fcn = std::make_unique<RooMinimizerFcn>(&function, this);
   }
   initMinimizerFcnDependentPart(function.defaultErrorLevel());
}

/// Initialize the part of the minimizer that is independent of the function to be minimized
void RooMinimizer::initMinimizerFirstPart()
{
   RooSentinel::activate();
   setMinimizerType("");
   _minuit2 = std::make_unique<Minuit2State>();

   _config.SetMinimizer(_cfg.minimizerType.c_str());
   setEps(1.0); // default tolerance
}

/// Initialize the part of the minimizer that is dependent on the function to be minimized
void RooMinimizer::initMinimizerFcnDependentPart(double defaultErrorLevel)
{
   // default max number of calls
   _config.MinimizerOptions().SetMaxIterations(500 * _fcn->getNDim());
   _config.MinimizerOptions().SetMaxFunctionCalls(500 * _fcn->getNDim());

   // Shut up for now
   setPrintLevel(-1);

   // Use +0.5 for 1-sigma errors
   setErrorLevel(defaultErrorLevel);

   // Declare our parameters to MINUIT
   _fcn->Synchronize(_config.ParamsSettings());

   // Now set default verbosity
   setPrintLevel(RooMsgService::instance().silentMode() ? -1 : 1);

   // Set user defined and default _fcn config
   setLogFile(_cfg.logf);

   // Likelihood holds information on offsetting in old style, so do not set here unless explicitly set by user
   if (_cfg.offsetting != -1) {
      setOffsetting(_cfg.offsetting);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooMinimizer::~RooMinimizer() = default;

////////////////////////////////////////////////////////////////////////////////
/// Change MINUIT strategy to istrat. Accepted codes
/// are 0,1,2 and represent MINUIT strategies for dealing
/// most efficiently with fast FCNs (0), expensive FCNs (2)
/// and 'intermediate' FCNs (1)

void RooMinimizer::setStrategy(int istrat)
{
   _config.MinimizerOptions().SetStrategy(istrat);
}

////////////////////////////////////////////////////////////////////////////////
/// Change maximum number of MINUIT iterations
/// (RooMinimizer default 500 * #%parameters)

void RooMinimizer::setMaxIterations(int n)
{
   _config.MinimizerOptions().SetMaxIterations(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Change maximum number of likelihood function class from MINUIT
/// (RooMinimizer default 500 * #%parameters)

void RooMinimizer::setMaxFunctionCalls(int n)
{
   _config.MinimizerOptions().SetMaxFunctionCalls(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the level for MINUIT error analysis to the given
/// value. This function overrides the default value
/// that is taken in the RooMinimizer constructor from
/// the defaultErrorLevel() method of the input function

void RooMinimizer::setErrorLevel(double level)
{
   _config.MinimizerOptions().SetErrorDef(level);
}

////////////////////////////////////////////////////////////////////////////////
/// Change MINUIT epsilon

void RooMinimizer::setEps(double eps)
{
   _config.MinimizerOptions().SetTolerance(eps);
}

////////////////////////////////////////////////////////////////////////////////
/// Enable internal likelihood offsetting for enhanced numeric precision

void RooMinimizer::setOffsetting(bool flag)
{
   _cfg.offsetting = flag;
   _fcn->setOffsetting(_cfg.offsetting);
}

////////////////////////////////////////////////////////////////////////////////
/// Choose the minimizer algorithm.
///
/// Passing an empty string selects the default minimizer type returned by
/// ROOT::Math::MinimizerOptions::DefaultMinimizerType().

void RooMinimizer::setMinimizerType(std::string const &type)
{
   _cfg.minimizerType = type.empty() ? ROOT::Math::MinimizerOptions::DefaultMinimizerType() : type;

   if ((_cfg.parallelize != 0) && _cfg.minimizerType != "Minuit2") {
      std::stringstream ss;
      ss << "In RooMinimizer::setMinimizerType: only Minuit2 is supported when not using classic function mode!";
      if (type.empty()) {
         ss << "\nPlease set it as your default minimizer via "
               "ROOT::Math::MinimizerOptions::SetDefaultMinimizer(\"Minuit2\").";
      }
      throw std::invalid_argument(ss.str());
   }
}

void RooMinimizer::determineStatus(bool fitterReturnValue)
{
   // Minuit-given status:
   _status = fitterReturnValue ? _result->fStatus : -1;

   // RooFit-based additional failed state information:
   if (evalCounter() <= _fcn->GetNumInvalidNLL()) {
      coutE(Minimization) << "RooMinimizer: all function calls during minimization gave invalid NLL values!"
                          << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Minimise the function passed in the constructor.
/// \param[in] type Type of fitter to use, e.g. "Minuit" "Minuit2". Passing an
///                 empty string will select the default minimizer type of the
///                 RooMinimizer, as returned by
///                 ROOT::Math::MinimizerOptions::DefaultMinimizerType().
/// \attention This overrides the default fitter of this RooMinimizer.
/// \param[in] alg  Fit algorithm to use. (Optional)
int RooMinimizer::minimize(const char *type, const char *alg)
{

   if (_cfg.timingAnalysis) {
#ifdef ROOFIT_MULTIPROCESS
      addParamsToProcessTimer();
#else
      throw std::logic_error("ProcessTimer requested, but multiprocessing is not supported on this platform.");
#endif
   }
   _fcn->Synchronize(_config.ParamsSettings());

   setMinimizerType(type);
   _config.SetMinimizer(_cfg.minimizerType.c_str(), alg);

   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      bool ret = fitFCN();
      determineStatus(ret);
   }
   profileStop();
   _fcn->BackProp();

   saveStatus("MINIMIZE", _status);

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute MIGRAD. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::migrad()
{
   return exec("migrad", "MIGRAD");
}

int RooMinimizer::exec(std::string const &algoName, std::string const &statusName)
{
   FreezeDisconnectedParametersRAII freeze(this, *_fcn);

   _fcn->Synchronize(_config.ParamsSettings());
   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      bool ret = false;
      if (algoName == "hesse") {
         // HESSE has a special entry point in the ROOT::Math::Fitter
         _config.SetMinimizer(_cfg.minimizerType.c_str());
         ret = calculateHessErrors();
      } else if (algoName == "minos") {
         // MINOS has a special entry point in the ROOT::Math::Fitter
         _config.SetMinimizer(_cfg.minimizerType.c_str());
         ret = calculateMinosErrors();
      } else {
         _config.SetMinimizer(_cfg.minimizerType.c_str(), algoName.c_str());
         ret = fitFCN();
      }
      determineStatus(ret);
   }
   profileStop();
   _fcn->BackProp();

   saveStatus(statusName.c_str(), _status);

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute HESSE. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::hesse()
{
   if (_result == nullptr) {
      coutW(Minimization) << "RooMinimizer::hesse: Error, run Migrad before Hesse!" << std::endl;
      _status = -1;
      return _status;
   }

   return exec("hesse", "HESSE");
}

////////////////////////////////////////////////////////////////////////////////
/// Execute MINOS. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::minos()
{
   if (_result == nullptr) {
      coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!" << std::endl;
      _status = -1;
      return _status;
   }

   return exec("minos", "MINOS");
}

////////////////////////////////////////////////////////////////////////////////
/// Execute MINOS for given list of parameters. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::minos(const RooArgSet &minosParamList)
{
   if (_result == nullptr) {
      coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!" << std::endl;
      _status = -1;
   } else if (!minosParamList.empty()) {
      FreezeDisconnectedParametersRAII freeze(this, *_fcn);

      _fcn->Synchronize(_config.ParamsSettings());
      profileStart();
      {
         auto ctx = makeEvalErrorContext();

         // get list of parameters for Minos
         std::vector<unsigned int> paramInd;
         RooArgList floatParams = _fcn->floatParams();
         for (RooAbsArg *arg : minosParamList) {
            RooAbsArg *par = floatParams.find(arg->GetName());
            if (par && !par->isConstant()) {
               int index = floatParams.index(par);
               paramInd.push_back(index);
            }
         }

         if (!paramInd.empty()) {
            // set the parameter indices
            _config.SetMinosErrors(paramInd);

            _config.SetMinimizer(_cfg.minimizerType.c_str());
            bool ret = calculateMinosErrors();
            determineStatus(ret);
            // to avoid that following minimization computes automatically the Minos errors
            _config.SetMinosErrors(false);
         }
      }
      profileStop();
      _fcn->BackProp();

      saveStatus("MINOS", _status);
   }

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute SEEK. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::seek()
{
   return exec("seek", "SEEK");
}

////////////////////////////////////////////////////////////////////////////////
/// Execute SIMPLEX. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::simplex()
{
   return exec("simplex", "SIMPLEX");
}

////////////////////////////////////////////////////////////////////////////////
/// Execute IMPROVE. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::improve()
{
   return exec("migradimproved", "IMPROVE");
}

////////////////////////////////////////////////////////////////////////////////
/// Change the MINUIT internal printing level

void RooMinimizer::setPrintLevel(int newLevel)
{
   _config.MinimizerOptions().SetPrintLevel(newLevel + 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Get the MINUIT internal printing level

int RooMinimizer::getPrintLevel()
{
   return _config.MinimizerOptions().PrintLevel() + 1;
}

////////////////////////////////////////////////////////////////////////////////
/// \deprecated Has no effect anymore. Functionality was removed in ROOT 6.42,
/// and this function is kept as an empty shell that does nothing (for API
/// compatibility between different ROOT versions).

void RooMinimizer::optimizeConst(int /*flag*/) {}

////////////////////////////////////////////////////////////////////////////////
/// Save and return a RooFitResult snapshot of current minimizer status.
/// This snapshot contains the values of all constant parameters,
/// the value of all floating parameters at RooMinimizer construction and
/// after the last MINUIT operation, the MINUIT status, variance quality,
/// EDM setting, number of calls with evaluation problems, the minimized
/// function value and the full correlation matrix.

RooFit::OwningPtr<RooFitResult> RooMinimizer::save(const char *userName, const char *userTitle)
{
   if (_result == nullptr) {
      coutW(Minimization) << "RooMinimizer::save: Error, run minimization before!" << std::endl;
      return nullptr;
   }

   std::string name = userName ? std::string{userName} : _fcn->getFunctionName();
   std::string title = userTitle ? std::string{userTitle} : _fcn->getFunctionTitle();
   auto fitRes = std::make_unique<RooFitResult>(name.c_str(), title.c_str());

   fitRes->setConstParList(_fcn->constParams());

   fitRes->setNumInvalidNLL(_fcn->GetNumInvalidNLL());

   fitRes->setStatus(_status);
   fitRes->setCovQual(_result->fCovStatus);
   fitRes->setMinNLL(_result->fVal - _fcn->getOffset());
   fitRes->setEDM(_result->fEdm);

   fitRes->setInitParList(_fcn->initFloatParams());
   fitRes->setFinalParList(_fcn->floatParams());

   if (!_extV) {
      fillCorrMatrix(*fitRes);
   } else {
      fitRes->setCovarianceMatrix(*_extV);
   }

   fitRes->setStatusHistory(_statusHistory);

   return RooFit::makeOwningPtr(std::move(fitRes));
}

namespace {

/// retrieve covariance matrix element
double covMatrix(std::vector<double> const &covMat, unsigned int i, unsigned int j)
{
   if (covMat.empty())
      return 0; // no matrix is available in case of non-valid fits
   return j < i ? covMat[j + i * (i + 1) / 2] : covMat[i + j * (j + 1) / 2];
}

/// retrieve correlation elements
double correlation(std::vector<double> const &covMat, unsigned int i, unsigned int j)
{
   if (covMat.empty())
      return 0; // no matrix is available in case of non-valid fits
   double tmp = covMatrix(covMat, i, i) * covMatrix(covMat, j, j);
   return tmp > 0 ? covMatrix(covMat, i, j) / std::sqrt(tmp) : 0;
}

} // namespace

void RooMinimizer::fillCorrMatrix(RooFitResult &fitRes)
{
   const std::size_t nParams = _fcn->getNDim();
   TMatrixDSym corrs(nParams);
   TMatrixDSym covs(nParams);
   std::vector<double> globalCC = _result->fGlobalCC;
   globalCC.resize(nParams); // pad with zeros
   for (std::size_t ic = 0; ic < nParams; ic++) {
      for (std::size_t ii = 0; ii < nParams; ii++) {
         corrs(ic, ii) = correlation(_result->fCovMatrix, ic, ii);
         covs(ic, ii) = covMatrix(_result->fCovMatrix, ic, ii);
      }
   }
   fitRes.fillCorrMatrix(globalCC, corrs, covs);
}

////////////////////////////////////////////////////////////////////////////////
/// Create and draw a TH2 with the error contours in the parameters `var1` and `var2`.
/// \param[in] var1 The first parameter (x axis).
/// \param[in] var2 The second parameter (y axis).
/// \param[in] n1 First contour.
/// \param[in] n2 Optional contour. 0 means don't draw.
/// \param[in] n3 Optional contour. 0 means don't draw.
/// \param[in] n4 Optional contour. 0 means don't draw.
/// \param[in] n5 Optional contour. 0 means don't draw.
/// \param[in] n6 Optional contour. 0 means don't draw.
/// \param[in] npoints Number of points for evaluating the contour.
///
/// Up to six contours can be drawn using the arguments `n1` to `n6` to request the desired
/// coverage in units of \f$ \sigma = n^2 \cdot \mathrm{ErrorDef} \f$.
/// See ROOT::Math::Minimizer::ErrorDef().

RooPlot *RooMinimizer::contour(RooRealVar &var1, RooRealVar &var2, double n1, double n2, double n3, double n4,
                               double n5, double n6, unsigned int npoints)
{
   RooArgList params = _fcn->floatParams();
   RooArgList paramSave;
   params.snapshot(paramSave);

   // Verify that both variables are floating parameters of PDF
   int index1 = params.index(&var1);
   if (index1 < 0) {
      coutE(Minimization) << "RooMinimizer::contour(" << GetName() << ") ERROR: " << var1.GetName()
                          << " is not a floating parameter of " << _fcn->getFunctionName() << std::endl;
      return nullptr;
   }

   int index2 = params.index(&var2);
   if (index2 < 0) {
      coutE(Minimization) << "RooMinimizer::contour(" << GetName() << ") ERROR: " << var2.GetName()
                          << " is not a floating parameter of PDF " << _fcn->getFunctionName() << std::endl;
      return nullptr;
   }

   // create and draw a frame
   RooPlot *frame = new RooPlot(var1, var2);

   // draw a point at the current parameter values
   TMarker *point = new TMarker(var1.getVal(), var2.getVal(), 8);
   frame->addObject(point);

   // check first if a minimization was done
   if (_result == nullptr) {
      coutW(Minimization) << "RooMinimizer::contour: Error, run Migrad before contours!" << std::endl;
      return frame;
   }

   // remember our original value of ERRDEF
   const double errdef = _config.MinimizerOptions().ErrorDef();

   // compute a contour at the given error level
   auto computeContour = [&](double up, double *xcoor, double *ycoor) {
      if (useMinuit2Directly()) {
         return minuit2Contour(index1, index2, npoints, up, xcoor, ycoor);
      }
      _minimizer->SetErrorDef(up);
      return _minimizer->Contour(index1, index2, npoints, xcoor, ycoor);
   };

   double n[6];
   n[0] = n1;
   n[1] = n2;
   n[2] = n3;
   n[3] = n4;
   n[4] = n5;
   n[5] = n6;

   auto operModeRAII = setOperModesDirty(_function);
   for (int ic = 0; ic < 6; ic++) {
      if (n[ic] > 0) {

         // calculate and draw the contour at the level corresponding to n-sigma
         std::vector<double> xcoor(npoints + 1);
         std::vector<double> ycoor(npoints + 1);
         bool ret = computeContour(n[ic] * n[ic] * errdef, xcoor.data(), ycoor.data());

         if (!ret) {
            coutE(Minimization) << "RooMinimizer::contour(" << GetName()
                                << ") ERROR: MINUIT did not return a contour graph for n=" << n[ic] << std::endl;
         } else {
            xcoor[npoints] = xcoor[0];
            ycoor[npoints] = ycoor[0];
            TGraph *graph = new TGraph(npoints + 1, xcoor.data(), ycoor.data());

            std::stringstream name;
            name << "contour_" << _fcn->getFunctionName() << "_n" << n[ic];
            graph->SetName(name.str().c_str());
            graph->SetLineStyle(ic + 1);
            graph->SetLineWidth(2);
            graph->SetLineColor(kBlue);
            frame->addObject(graph, "L");
         }
      }
   }

   // restore the original ERRDEF
   if (useMinuit2Directly()) {
      setMinuit2ErrorDef(errdef);
   } else {
      _minimizer->SetErrorDef(errdef);
   }

   // restore parameter values
   params.assign(paramSave);

   return frame;
}

////////////////////////////////////////////////////////////////////////////////
/// Add parameters in metadata field to process timer

void RooMinimizer::addParamsToProcessTimer()
{
#ifdef ROOFIT_MULTIPROCESS
   // parameter indices for use in timing heat matrix
   std::vector<std::string> parameter_names;
   for (RooAbsArg *parameter : _fcn->floatParams()) {
      parameter_names.push_back(parameter->GetName());
      if (_cfg.verbose) {
         coutI(Minimization) << "parameter name: " << parameter_names.back() << std::endl;
      }
   }
   RooFit::MultiProcess::ProcessTimer::add_metadata(parameter_names);
#else
   coutI(Minimization) << "Not adding parameters to processtimer because multiprocessing is not enabled." << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Start profiling timer

void RooMinimizer::profileStart()
{
   if (_cfg.profile) {
      _timer.Start();
      _cumulTimer.Start(_profileStart ? false : true);
      _profileStart = true;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Stop profiling timer and report results of last session

void RooMinimizer::profileStop()
{
   if (_cfg.profile) {
      _timer.Stop();
      _cumulTimer.Stop();
      coutI(Minimization) << "Command timer: ";
      _timer.Print();
      coutI(Minimization) << "Session timer: ";
      _cumulTimer.Print();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Apply results of given external covariance matrix. i.e. propagate its errors
/// to all RRV parameter representations and give this matrix instead of the
/// HESSE matrix at the next save() call

void RooMinimizer::applyCovarianceMatrix(TMatrixDSym const &V)
{
   _extV.reset(static_cast<TMatrixDSym *>(V.Clone()));
   _fcn->ApplyCovarianceMatrix(*_extV);
}

RooFit::OwningPtr<RooFitResult> RooMinimizer::lastMinuitFit()
{
   // Import the results of the last fit performed, interpreting
   // the fit parameters as the given varList of parameters.

   if (_result == nullptr) {
      oocoutE(nullptr, InputArguments) << "RooMinimizer::save: Error, run minimization before!" << std::endl;
      return nullptr;
   }

   auto res = std::make_unique<RooFitResult>("lastMinuitFit", "Last MINUIT fit");

   // Extract names of fit parameters
   // and construct corresponding RooRealVars
   RooArgList constPars("constPars");
   RooArgList floatPars("floatPars");

   const RooArgList floatParsFromFcn = _fcn->floatParams();

   for (unsigned int i = 0; i < _fcn->getNDim(); ++i) {

      TString varName(floatParsFromFcn.at(i)->GetName());
      bool isConst(_result->isParameterFixed(i));

      double xlo = _config.ParSettings(i).LowerLimit();
      double xhi = _config.ParSettings(i).UpperLimit();
      double xerr = _result->error(i);
      double xval = _result->fParams[i];

      std::unique_ptr<RooRealVar> var;

      if ((xlo < xhi) && !isConst) {
         var = std::make_unique<RooRealVar>(varName, varName, xval, xlo, xhi);
      } else {
         var = std::make_unique<RooRealVar>(varName, varName, xval);
      }
      var->setConstant(isConst);

      if (isConst) {
         constPars.addOwned(std::move(var));
      } else {
         var->setError(xerr);
         floatPars.addOwned(std::move(var));
      }
   }

   res->setConstParList(constPars);
   res->setInitParList(floatPars);
   res->setFinalParList(floatPars);
   res->setMinNLL(_result->fVal);
   res->setEDM(_result->fEdm);
   res->setCovQual(_result->fCovStatus);
   res->setStatus(_result->fStatus);
   fillCorrMatrix(*res);

   return RooFit::makeOwningPtr(std::move(res));
}

/// Try to recover from invalid function values. When invalid function values
/// are encountered, a penalty term is returned to the minimiser to make it
/// back off. This sets the strength of this penalty. \note A strength of zero
/// is equivalent to a constant penalty (= the gradient vanishes, ROOT < 6.24).
/// Positive values lead to a gradient pointing away from the undefined
/// regions. Use ~10 to force the minimiser away from invalid function values.
void RooMinimizer::setRecoverFromNaNStrength(double strength)
{
   _cfg.recoverFromNaN = strength;
}

bool RooMinimizer::setLogFile(const char *logf)
{
   _cfg.logf = logf;
   return _cfg.logf ? _fcn->SetLogFile(_cfg.logf) : false;
}

int RooMinimizer::evalCounter() const
{
   return _fcn->evalCounter();
}
void RooMinimizer::zeroEvalCount()
{
   _fcn->zeroEvalCount();
}

int RooMinimizer::getNPar() const
{
   return _fcn->getNDim();
}

std::ofstream *RooMinimizer::logfile()
{
   return _fcn->GetLogFile();
}
double &RooMinimizer::maxFCN()
{
   return _fcn->GetMaxFCN();
}
double &RooMinimizer::fcnOffset() const
{
   return _fcn->getOffset();
}

std::unique_ptr<RooAbsReal::EvalErrorContext> RooMinimizer::makeEvalErrorContext() const
{
   RooAbsReal::clearEvalErrorLog();
   // If evaluation error printing is disabled, we don't need to collect the
   // errors and only need to count them. This significantly reduces the
   // performance overhead when having evaluation errors.
   auto m = _cfg.printEvalErrors < 0 ? RooAbsReal::CountErrors : RooAbsReal::CollectErrors;
   return std::make_unique<RooAbsReal::EvalErrorContext>(m);
}

bool RooMinimizer::fitFCN()
{
   // fit a user provided FCN function
   // create fit parameter settings

   auto operModeRAII = setOperModesDirty(_function);

   // Check number of parameters
   unsigned int npar = getNPar();
   if (npar == 0) {
      coutE(Minimization) << "RooMinimizer::fitFCN(): FCN function has zero parameters" << std::endl;
      return false;
   }

   // initiate the minimizer
   initMinimizer();

   // Identify floating RooCategory parameters
   RooArgSet floatingCats;
   for (auto arg : _fcn->allParams()) {
      if (arg->isCategory() && !arg->isConstant())
         floatingCats.add(*arg);
   }

   std::vector<RooCategory *> pdfIndices;
   for (auto *arg : floatingCats) {
      if (auto *cat = dynamic_cast<RooCategory *>(arg))
         pdfIndices.push_back(cat);
   }

   const size_t nPdfs = pdfIndices.size();

   // Identify floating continuous parameters (RooRealVar)
   RooArgSet floatReals;
   for (auto arg : _fcn->allParams()) {
      if (!arg->isCategory() && !arg->isConstant())
         floatReals.add(*arg);
   }

   if (nPdfs == 0) {
      coutI(Minimization) << "[fitFCN] No discrete parameters, performing continuous minimization only" << std::endl;
      FreezeDisconnectedParametersRAII freeze(this, *_fcn);
      bool isValid = runMinimizer();
      if (isValid)
         updateFitConfig();
      return isValid;
   }

   // set also new parameter values and errors in FitConfig
   // Prepare discrete indices
   std::vector<int> maxIndices;
   for (auto *cat : pdfIndices)
      maxIndices.push_back(cat->size());

   std::set<std::vector<int>> tried;
   std::map<std::vector<int>, double> nllMap;
   std::vector<int> bestIndices(nPdfs, 0);
   double bestNLL = 1e30;

   bool improved = true;
   while (improved) {
      improved = false;
      auto combos = generateOrthogonalCombinations(maxIndices);
      reorderCombinations(combos, maxIndices, bestIndices);

      for (const auto &combo : combos) {
         if (tried.count(combo))
            continue;

         for (size_t i = 0; i < nPdfs; ++i)
            pdfIndices[i]->setIndex(combo[i]);

         // Freeze categories during continuous minimization
         std::vector<bool> wasConst(nPdfs);
         for (size_t i = 0; i < nPdfs; ++i) {
            wasConst[i] = pdfIndices[i]->isConstant();
            pdfIndices[i]->setConstant(true);
         }
         FreezeDisconnectedParametersRAII freeze(this, *_fcn);
         runMinimizer();

         for (size_t i = 0; i < nPdfs; ++i)
            pdfIndices[i]->setConstant(wasConst[i]);

         double val = _result->fVal;
         tried.insert(combo);
         nllMap[combo] = val;

         if (val < bestNLL) {
            bestNLL = val;
            bestIndices = combo;
            improved = true;
         }
      }
   }

   for (size_t i = 0; i < nPdfs; ++i) {
      pdfIndices[i]->setIndex(bestIndices[i]);
   }

   FreezeDisconnectedParametersRAII freeze(this, *_fcn);
   runMinimizer();

   coutI(Minimization) << "All NLL Values per Combination:" << std::endl;
   for (const auto &entry : nllMap) {
      const auto &combo = entry.first;
      double val = entry.second;

      std::stringstream ss;
      ss << "Combo: [";
      for (size_t i = 0; i < combo.size(); ++i) {
         ss << combo[i];
         if (i + 1 < combo.size())
            ss << ", ";
      }
      ss << "], NLL: " << val;

      coutI(Minimization) << ss.str() << std::endl;
   }

   std::stringstream ssBest;
   ssBest << "DP Best Indices: [";
   for (size_t i = 0; i < bestIndices.size(); ++i) {
      ssBest << bestIndices[i];
      if (i + 1 < bestIndices.size())
         ssBest << ", ";
   }
   ssBest << "], NLL = " << bestNLL;

   coutI(Minimization) << ssBest.str() << std::endl;

   _result->fValid = true;
   updateFitConfig();

   return true;
}

/// Run the minimization algorithm from the current minimizer state and store
/// the outcome in the internal fit result. Returns whether a valid minimum was
/// found.
bool RooMinimizer::runMinimizer()
{
   bool isValid = useMinuit2Directly() ? minuit2Minimize() : _minimizer->Minimize();
   if (!_result)
      _result = std::make_unique<FitResult>();
   fillResult(isValid);
   return isValid;
}

bool RooMinimizer::calculateHessErrors()
{
   // compute the Hesse errors according to configuration
   // set in the parameters and append value in fit result

   auto operModeRAII = setOperModesDirty(_function);

   bool ret = false;
   if (useMinuit2Directly()) {
      ret = minuit2Hesse();
   } else {
      // update  minimizer (recreate if not done or if name has changed
      if (!updateMinimizerOptions()) {
         coutE(Minimization) << "RooMinimizer::calculateHessErrors() Error re-initializing the minimizer" << std::endl;
         return false;
      }
      ret = _minimizer->Hesse();
   }
   if (!ret)
      coutE(Minimization) << "RooMinimizer::calculateHessErrors() Error when calculating Hessian" << std::endl;

   // update minimizer results with what comes out from Hesse
   // in case is empty - create from a FitConfig
   if (_result->fParams.empty())
      _result = std::make_unique<FitResult>(_config);

   // re-give a minimizer instance in case it has been changed
   ret |= update(ret);

   // set also new errors in FitConfig
   if (ret)
      updateFitConfig();

   return ret;
}

bool RooMinimizer::calculateMinosErrors()
{
   // compute the Minos errors according to configuration
   // set in the parameters and append value in fit result
   // normally Minos errors are computed just after the minimization
   // (in DoMinimization) aftewr minimizing if the
   //  FitConfig::MinosErrors() flag is set

   auto operModeRAII = setOperModesDirty(_function);

   // update  minimizer (but cannot re-create in this case). Must use an existing one
   if (!useMinuit2Directly() && !updateMinimizerOptions(false)) {
      coutE(Minimization) << "RooMinimizer::calculateMinosErrors() Error re-initializing the minimizer" << std::endl;
      return false;
   }

   const std::vector<unsigned int> &ipars = _config.MinosParams();
   unsigned int n = (!ipars.empty()) ? ipars.size() : _fcn->getNDim();
   bool ok = false;

   int iparNewMin = 0;
   int iparMax = n;
   int iter = 0;
   // rerun minos for the parameters run before a new Minimum has been found
   do {
      if (iparNewMin > 0)
         coutI(Minimization) << "RooMinimizer::calculateMinosErrors() Run again Minos for some parameters because a "
                                "new Minimum has been found"
                             << std::endl;
      iparNewMin = 0;
      for (int i = 0; i < iparMax; ++i) {
         double elow, eup;
         unsigned int index = (!ipars.empty()) ? ipars[i] : i;
         bool ret = false;
         int minosStatus = 0;
         if (useMinuit2Directly()) {
            ret = minuit2Minos(index, elow, eup);
            minosStatus = _minuit2->minosStatus;
         } else {
            ret = _minimizer->GetMinosError(index, elow, eup);
            minosStatus = _minimizer->MinosStatus();
         }
         // flags case when a new minimum has been found
         if ((minosStatus & 8) != 0) {
            iparNewMin = i;
         }
         if (ret)
            _result->fMinosErrors.emplace(index, std::make_pair(elow, eup));
         ok |= ret;
      }

      iparMax = iparNewMin;
      iter++; // to avoid infinite looping
   } while (iparNewMin > 0 && iter < 10);
   if (!ok) {
      coutE(Minimization)
         << "RooMinimizer::calculateMinosErrors() Minos error calculation failed for all the selected parameters"
         << std::endl;
   }

   // re-give a minimizer instance in case it has been changed
   // but maintain previous valid status. Do not set result to false if minos failed
   ok &= update(_result->fValid);

   return ok;
}

void RooMinimizer::initMinimizer()
{
   if (useMinuit2Directly()) {
      initMinuit2();
      return;
   }
   _minimizer = std::unique_ptr<ROOT::Math::Minimizer>(_config.CreateMinimizer());
   _fcn->initMinimizer(*_minimizer, this);
   _minimizer->SetVariables(_config.ParamsSettings().begin(), _config.ParamsSettings().end());

   if (_cfg.setInitialCovariance) {
      std::vector<double> v;
      for (std::size_t i = 0; i < _fcn->getNDim(); ++i) {
         RooRealVar &param = _fcn->floatableParam(i);
         v.push_back(param.getError() * param.getError());
      }
      _minimizer->SetCovarianceDiag(v, v.size());
   }
}

bool RooMinimizer::updateMinimizerOptions(bool canDifferentMinim)
{
   // update minimizer options when re-doing a Fit or computing Hesse or Minos errors

   // create a new minimizer if it is different type
   // minimizer type string stored in FitResult is "minimizer name" + " / " + minimizer algo
   std::string newMinimType = _config.MinimizerName();
   if (_minimizer && _result && newMinimType != _result->fMinimType) {
      // if a different minimizer is allowed (e.g. when calling Hesse)
      if (canDifferentMinim) {
         std::string msg = "Using now " + newMinimType;
         coutI(Minimization) << "RooMinimizer::updateMinimizerOptions(): " << msg << std::endl;
         initMinimizer();
      } else {
         std::string msg = "Cannot change minimizer. Continue using " + _result->fMinimType;
         coutW(Minimization) << "RooMinimizer::updateMinimizerOptions() " << msg << std::endl;
      }
   }

   // create minimizer if it was not done before
   if (!_minimizer) {
      initMinimizer();
   }

   // set new minimizer options (but not functions and parameters)
   _minimizer->SetOptions(_config.MinimizerOptions());
   return true;
}

void RooMinimizer::updateFitConfig()
{
   // update the fit configuration after a fit using the obtained result
   if (_result->fParams.empty() || !_result->fValid)
      return;
   for (unsigned int i = 0; i < _config.NPar(); ++i) {
      ROOT::Fit::ParameterSettings &par = _config.ParSettings(i);
      par.SetValue(_result->fParams[i]);
      if (_result->error(i) > 0)
         par.SetStepSize(_result->error(i));
   }
}

RooMinimizer::FitResult::FitResult(const ROOT::Fit::FitConfig &fconfig)
   : fStatus(-99), // use this special convention to flag it when printing result
     fCovStatus(0),
     fParams(fconfig.NPar()),
     fErrors(fconfig.NPar())
{
   // create a Fit result from a fit config (i.e. with initial parameter values
   // and errors equal to step values
   // The model function is NULL in this case

   // set minimizer type and algorithm
   fMinimType = fconfig.MinimizerType();
   // append algorithm name for minimizer that support it
   if ((fMinimType.find("Fumili") == std::string::npos) && (fMinimType.find("GSLMultiFit") == std::string::npos)) {
      if (!fconfig.MinimizerAlgoType().empty())
         fMinimType += " / " + fconfig.MinimizerAlgoType();
   }

   // get parameter values and errors (step sizes)
   for (unsigned int i = 0; i < fconfig.NPar(); ++i) {
      const ROOT::Fit::ParameterSettings &par = fconfig.ParSettings(i);
      fParams[i] = par.Value();
      fErrors[i] = par.StepSize();
      if (par.IsFixed())
         fFixedParams[i] = true;
   }
}

void RooMinimizer::fillResult(bool isValid)
{
   if (useMinuit2Directly()) {
      fillResultFromMinuit2(isValid);
      return;
   }
   ROOT::Math::Minimizer &min = *_minimizer;
   ROOT::Fit::FitConfig const &fconfig = _config;

   // Fill the FitResult after minimization using result from Minimizers

   _result->fValid = isValid;
   _result->fStatus = min.Status();
   _result->fCovStatus = min.CovMatrixStatus();
   _result->fVal = min.MinValue();
   _result->fEdm = min.Edm();

   _result->fMinimType = fconfig.MinimizerName();

   const unsigned int npar = min.NDim();
   if (npar == 0)
      return;

   if (min.X())
      _result->fParams = std::vector<double>(min.X(), min.X() + npar);
   else {
      // case minimizer does not provide minimum values (it failed) take from configuration
      _result->fParams.resize(npar);
      for (unsigned int i = 0; i < npar; ++i) {
         _result->fParams[i] = (fconfig.ParSettings(i).Value());
      }
   }

   // check for fixed or limited parameters
   for (unsigned int ipar = 0; ipar < npar; ++ipar) {
      if (fconfig.ParSettings(ipar).IsFixed())
         _result->fFixedParams[ipar] = true;
   }

   // fill error matrix
   // if minimizer provides error provides also error matrix
   // clear in case of re-filling an existing result
   _result->fCovMatrix.clear();
   _result->fGlobalCC.clear();

   if (min.Errors() != nullptr) {
      updateErrors();
   }
}

bool RooMinimizer::update(bool isValid)
{
   if (useMinuit2Directly()) {
      fillResultFromMinuit2(isValid);
      return true;
   }
   ROOT::Math::Minimizer &min = *_minimizer;
   ROOT::Fit::FitConfig const &fconfig = _config;

   // update fit result with new status from minimizer
   // ncalls if it is not zero is used instead of value from minimizer

   // in case minimizer changes
   _result->fMinimType = fconfig.MinimizerName();

   const std::size_t npar = _result->fParams.size();

   _result->fValid = isValid;
   // update minimum value
   _result->fVal = min.MinValue();
   _result->fEdm = min.Edm();
   _result->fStatus = min.Status();
   _result->fCovStatus = min.CovMatrixStatus();

   // copy parameter value and errors
   std::copy(min.X(), min.X() + npar, _result->fParams.begin());

   if (min.Errors() != nullptr) {
      updateErrors();
   }
   return true;
}

void RooMinimizer::updateErrors()
{
   ROOT::Math::Minimizer &min = *_minimizer;
   const std::size_t npar = _result->fParams.size();

   _result->fErrors.resize(npar);
   std::copy(min.Errors(), min.Errors() + npar, _result->fErrors.begin());
   _result->fGlobalCC = min.GlobalCC();

   if (_result->fCovStatus != 0) {

      // update error matrix
      unsigned int r = npar * (npar + 1) / 2;
      _result->fCovMatrix.resize(r);
      unsigned int l = 0;
      for (unsigned int i = 0; i < npar; ++i) {
         for (unsigned int j = 0; j <= i; ++j)
            _result->fCovMatrix[l++] = min.CovMatrix(i, j);
      }
   }
   // minos errors are set separately when calling Fitter::CalculateMinosErrors()
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the Minuit2 parameter state from the parameter settings, and
/// forget about any previous function minimum.

void RooMinimizer::initMinuit2()
{
   using ROOT::Minuit2::MnUserParameterState;

   _minuit2->state = MnUserParameterState{};
   _minuit2->minimum.reset();
   _minuit2->status = 0;
   _minuit2->minosStatus = -1;

   MnUserParameterState &state = _minuit2->state;

   for (ROOT::Fit::ParameterSettings const &par : _config.ParamsSettings()) {
      if (par.IsFixed()) {
         // Fixed parameters still need a step size, otherwise Minuit2 would
         // treat them as constants that can't be released anymore.
         const double step = par.Value() != 0 ? 0.1 * std::abs(par.Value()) : 0.1;
         state.Add(par.Name(), par.Value(), step);
         state.Fix(par.Name());
         continue;
      }
      if (par.StepSize() <= 0) {
         // A parameter without a valid step size is treated as a constant.
         state.Add(par.Name(), par.Value());
      } else {
         state.Add(par.Name(), par.Value(), par.StepSize());
      }
      if (par.IsDoubleBound()) {
         state.SetLimits(par.Name(), par.LowerLimit(), par.UpperLimit());
      } else if (par.HasLowerLimit()) {
         state.SetLowerLimit(par.Name(), par.LowerLimit());
      } else if (par.HasUpperLimit()) {
         state.SetUpperLimit(par.Name(), par.UpperLimit());
      }
   }

   if (_cfg.setInitialCovariance) {
      // Diagonal covariance matrix from the parameter errors, in the packed
      // lower-triangular storage of MnUserCovariance.
      const unsigned int n = _fcn->getNDim();
      ROOT::Minuit2::MnUserCovariance cov{n};
      for (unsigned int i = 0; i < n; ++i) {
         const double err = _fcn->floatableParam(i).getError();
         cov(i, i) = err * err;
      }
      state.AddCovariance(cov);
   }

   _minuit2->initialized = true;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the error definition on the function and on the function minimum, if
/// there is one.

void RooMinimizer::setMinuit2ErrorDef(double up)
{
   _fcn->SetErrorDef(up);
   if (_minuit2->minimum && _minuit2->minimum->Up() != up) {
      _minuit2->minimum->SetErrorDef(up);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Run the configured Minuit2 minimization algorithm, starting from the current
/// parameter state. Returns whether a valid minimum was found.

bool RooMinimizer::minuit2Minimize()
{
   using namespace ROOT::Minuit2;

   ROOT::Math::MinimizerOptions const &options = _config.MinimizerOptions();

   if (!_minuit2->initialized) {
      initMinuit2();
   }

   auto minimizer = makeMinuit2Minimizer(options.MinimizerAlgorithm());
   if (!minimizer) {
      coutE(Minimization) << "RooMinimizer::minimize: the Minuit2 algorithm \"" << options.MinimizerAlgorithm()
                          << "\" is not supported by RooFit" << std::endl;
      return false;
   }

   MnUserParameterState &state = _minuit2->state;

   // delete result of previous minimization
   _minuit2->minimum.reset();

   const unsigned int maxfcn = options.MaxFunctionCalls();
   const double tol = options.Tolerance();
   const int printLevel = options.PrintLevel();
   _fcn->SetErrorDef(options.ErrorDef());

   if (printLevel >= 1) {
      // print the real number of maxfcn used (defined in ModularFunctionMinimizer)
      unsigned int maxfcnUsed = maxfcn;
      if (maxfcnUsed == 0) {
         const unsigned int nvar = state.VariableParameters();
         maxfcnUsed = 200 + 100 * nvar + 5 * nvar * nvar;
      }
      std::cout << "RooMinimizer: Minuit2 minimize with max-calls " << maxfcnUsed << " convergence for edm < " << tol
                << " strategy " << options.Strategy() << std::endl;
   }

   minimizer->Builder().SetPrintLevel(printLevel);
   Minuit2PrintLevelRAII printLevelRAII{printLevel};

   if (options.Precision() > 0) {
      state.SetPrecision(options.Precision());
   }

   if (ROOT::Math::IOptions const *minuit2Opt = minuit2ExtraOptions(options)) {
      int storageLevel = 1;
      if (minuit2Opt->GetValue("StorageLevel", storageLevel)) {
         minimizer->Builder().SetStorageLevel(storageLevel);
      }
      if (printLevel > 0) {
         std::cout << "RooMinimizer: Minuit2 - Changing default options" << std::endl;
         minuit2Opt->Print();
      }
   }

   const MnStrategy strategy = makeMinuit2Strategy(options);

   _minuit2->minimum = std::make_unique<FunctionMinimum>(minimizer->Minimize(*_fcn, state, strategy, maxfcn, tol));
   FunctionMinimum const &minimum = *_minuit2->minimum;

   // copy minimum state (parameter values and errors)
   state = minimum.UserState();

   // Determine the status code, with the same convention as Minuit2Minimizer.
   int &status = _minuit2->status;
   status = 0;
   std::string txt;
   if (!minimum.HasPosDefCovar()) {
      // this happens normally when Hesse failed
      // it can happen in case MnSeed failed (see ROOT-9522)
      txt = "Covar is not pos def";
      status = 5;
   }
   if (minimum.HasMadePosDefCovar()) {
      txt = "Covar was made pos def";
      status = 1;
   }
   if (minimum.HesseFailed()) {
      txt = "Hesse is not valid";
      status = 2;
   }
   if (minimum.IsAboveMaxEdm()) {
      txt = "Edm is above max";
      status = 3;
   }
   if (minimum.HasReachedCallLimit()) {
      txt = "Reached call limit";
      status = 4;
   }

   MnPrint print("RooMinimizer::minimize", printLevel);
   const bool validMinimum = minimum.IsValid();
   if (validMinimum) {
      // print a warning message in case something is not ok
      if (status != 0 && printLevel > 0)
         print.Warn(txt);
   } else {
      // minimum is not valid when state is not valid and edm is over max or has passed call limits
      if (status == 0) {
         // this should not happen
         txt = "unknown failure";
         status = 6;
      }
      print.Warn("Minimization did NOT converge,", txt);
   }

   if (printLevel >= 1) {
      std::cout << "RooMinimizer: Minuit2 " << (validMinimum ? "valid" : "invalid") << " minimum - status = " << status
                << std::endl;
      const int prec = std::cout.precision(18);
      std::cout << "FVAL  = " << state.Fval() << std::endl;
      std::cout << "Edm   = " << state.Edm() << std::endl;
      std::cout.precision(prec);
      std::cout << "Nfcn  = " << state.NFcn() << std::endl;
      if (validMinimum) {
         for (MinuitParameter const &par : state.MinuitParameters()) {
            std::cout << par.Name() << "\t  = " << par.Value() << "\t ";
            if (par.IsFixed())
               std::cout << "(fixed)" << std::endl;
            else if (par.IsConst())
               std::cout << "(const)" << std::endl;
            else if (par.HasLimits())
               std::cout << "+/-  " << par.Error() << "\t(limited)" << std::endl;
            else
               std::cout << "+/-  " << par.Error() << std::endl;
         }
      }
   }

   return validMinimum;
}

////////////////////////////////////////////////////////////////////////////////
/// Run Minuit2's HESSE. If a function minimum from a previous minimization
/// exists, it is updated with the result. Returns whether a valid covariance
/// matrix was obtained.

bool RooMinimizer::minuit2Hesse()
{
   using namespace ROOT::Minuit2;

   ROOT::Math::MinimizerOptions const &options = _config.MinimizerOptions();

   if (!_minuit2->initialized) {
      initMinuit2();
   }

   const unsigned int maxfcn = options.MaxFunctionCalls();
   const int printLevel = options.PrintLevel();

   MnPrint print("RooMinimizer::hesse", printLevel);
   print.Info("Using max-calls", maxfcn);

   Minuit2PrintLevelRAII printLevelRAII{printLevel};

   MnUserParameterState &state = _minuit2->state;
   if (options.Precision() > 0) {
      state.SetPrecision(options.Precision());
   }

   setMinuit2ErrorDef(options.ErrorDef());

   MnHesse hesse(makeMinuit2Strategy(options));

   if (_minuit2->minimum) {
      // run hesse and function minimum will be updated with Hesse result
      hesse(*_fcn, *_minuit2->minimum, maxfcn);
      state = _minuit2->minimum->UserState();
   } else {
      // run Hesse on point stored in current state (independent of function minimum validity)
      state = hesse(*_fcn, state, maxfcn);
   }

   if (printLevel >= 3) {
      std::cout << "RooMinimizer::hesse - State returned from Hesse " << std::endl;
      std::cout << state << std::endl;
   }

   const int covStatus = state.CovarianceStatus();
   std::string covStatusType = "not valid";
   if (covStatus == 1)
      covStatusType = "approximate";
   if (covStatus == 2)
      covStatusType = "full but made positive defined";
   if (covStatus == 3)
      covStatusType = "accurate";
   if (covStatus == 0)
      covStatusType = "full but not positive defined";

   if (!state.HasCovariance()) {
      // if false means error is not valid and this is due to a failure in Hesse
      // update minimizer error status
      int hstatus = 4;
      // information on error state can be retrieved only if the minimum is available
      if (_minuit2->minimum) {
         if (_minuit2->minimum->Error().HesseFailed())
            hstatus = 1;
         if (_minuit2->minimum->Error().InvertFailed())
            hstatus = 2;
         else if (!(_minuit2->minimum->Error().IsPosDef()))
            hstatus = 3;
      }

      print.Warn("Hesse failed - matrix is", covStatusType);
      print.Warn(hstatus);

      _minuit2->status += 100 * hstatus;
      return false;
   }

   print.Info("Hesse is valid - matrix is", covStatusType);

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Run Minuit2's MINOS for the parameter with the given index. If a new
/// minimum is found, the minimization is repeated from there and MINOS is run
/// again. Returns whether both errors are valid.

bool RooMinimizer::minuit2Minos(unsigned int index, double &errLow, double &errUp)
{
   using namespace ROOT::Minuit2;

   errLow = 0;
   errUp = 0;

   const int printLevel = _config.MinimizerOptions().PrintLevel();
   MnPrint print("RooMinimizer::minos", printLevel);

   if (!_minuit2->minimum) {
      print.Error("Failed - no function minimum existing");
      return false;
   }

   MnUserParameterState &state = _minuit2->state;

   // need to know if parameter is const or fixed
   if (isFixedOrConst(state.Parameter(index))) {
      return false;
   }

   if (!_minuit2->minimum->IsValid()) {
      print.Error("Failed - invalid function minimum");
      return false;
   }

   setMinuit2ErrorDef(_config.MinimizerOptions().ErrorDef());

   int mstatus = runMinuit2Minos(index, errLow, errUp);

   // run again the Minimization in case of a new minimum
   // bit 8 is set
   if ((mstatus & 8) != 0) {
      print.Info([&](std::ostream &os) {
         os << "Found a new minimum: run again the Minimization starting from the new point";
         os << "\nFVAL  = " << state.Fval();
         for (MinuitParameter const &par : state.MinuitParameters()) {
            os << '\n' << par.Name() << "\t  = " << par.Value();
         }
      });
      // release parameter that was fixed in the returned state from Minos
      state.Release(index);
      if (!minuit2Minimize())
         return false;
      // run again Minos from new Minimum (also lower error needs to be re-computed)
      print.Info("Run now again Minos from the new found Minimum");
      mstatus = runMinuit2Minos(index, errLow, errUp);

      // do not reset new minimum bit to flag for other parameters
      mstatus |= 8;
   }

   _minuit2->status += 10 * mstatus;
   _minuit2->minosStatus = mstatus;

   return ((mstatus & 1) == 0) && ((mstatus & 2) == 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Run MINOS once for the parameter with the given index and return the MINOS
/// status bits, with the same convention as Minuit2Minimizer:
///   bit 1: lower error invalid, bit 2: upper error invalid, bit 4: invalid
///   because the maximum number of function calls was reached, bit 8: invalid
///   because a new minimum was found, bit 16: parameter is at a limit.

int RooMinimizer::runMinuit2Minos(unsigned int index, double &errLow, double &errUp)
{
   using namespace ROOT::Minuit2;

   ROOT::Math::MinimizerOptions const &options = _config.MinimizerOptions();
   const int printLevel = options.PrintLevel();
   MnPrint print("RooMinimizer::minos", printLevel);

   Minuit2PrintLevelRAII printLevelRAII{printLevel};

   MnUserParameterState &state = _minuit2->state;
   if (options.Precision() > 0) {
      state.SetPrecision(options.Precision());
   }

   // Like Minuit2Minimizer, use the default strategy for Minos.
   MnMinos minos(*_fcn, *_minuit2->minimum);

   const unsigned int maxfcn = options.MaxFunctionCalls();
   // Tolerance for the migrad calls inside Minos. Cut off too small values,
   // which are not needed.
   const double tol = std::max(options.Tolerance(), 0.01);

   const char *parName = state.Name(index);

   if (printLevel >= 1) {
      // get the real number of maxfcn used (defined in MnMinos) to be printed
      unsigned int maxfcnUsed = maxfcn;
      if (maxfcnUsed == 0) {
         const unsigned int nvar = state.VariableParameters();
         maxfcnUsed = 2 * (nvar + 1) * (200 + 100 * nvar + 5 * nvar * nvar);
      }
      std::cout << "RooMinimizer::minos - Run MINOS for parameter #" << index << " : " << parName << " using max-calls "
                << maxfcnUsed << ", tolerance " << tol << std::endl;
   }

   const MinosError me = minos.Minos(index, maxfcn, tol);

   // Note that the only invalid condition can happen when the (npar-1) minimization fails
   // The error is also invalid when the maximum number of calls is reached or a new function minimum is found
   // in case of the parameter at the limit the error is not invalid.
   // When the error is invalid the returned error is the Hessian error.
   if (!me.LowerValid()) {
      print.Warn("Invalid lower error for parameter", parName);
   }
   if (!me.UpperValid()) {
      print.Warn("Invalid upper error for parameter", parName);
   }
   if (me.AtLowerLimit()) {
      print.Warn("Lower error for parameter", parName, "is at the Lower limit!");
   }
   if (me.AtUpperLimit()) {
      print.Warn("Upper error for parameter", parName, "is at the Upper limit!");
   }
   if (me.AtLowerMaxFcn()) {
      print.Warn("Maximum number of function calls exceeded when running for lower error for parameter", parName);
   }
   if (me.AtUpperMaxFcn()) {
      print.Warn("Maximum number of function calls exceeded when running for upper error for parameter", parName);
   }
   if (me.LowerNewMin()) {
      print.Warn("New Minimum found while running Minos for lower error for parameter", parName);
   }
   if (me.UpperNewMin()) {
      print.Warn("New Minimum found while running Minos for upper error for parameter", parName);
   }
   if (printLevel >= 1) {
      if (me.LowerValid())
         std::cout << "Minos: Lower error for parameter " << parName << "  :  " << me.Lower() << std::endl;
      if (me.UpperValid())
         std::cout << "Minos: Upper error for parameter " << parName << "  :  " << me.Upper() << std::endl;
   }

   int mstatus = 0;
   if (!me.LowerValid()) {
      mstatus |= 1;
      if (me.AtLowerMaxFcn())
         mstatus |= 4;
      if (me.LowerNewMin())
         mstatus |= 8;
   }
   if (!me.UpperValid()) {
      mstatus |= 2;
      if (me.AtUpperMaxFcn())
         mstatus |= 4;
      if (me.UpperNewMin())
         mstatus |= 8;
   }
   if (me.AtUpperLimit() || me.AtLowerLimit())
      mstatus |= 16;

   errLow = me.Lower();
   errUp = me.Upper();

   // in case of new minimum found update also the minimum state
   if (me.LowerNewMin() && me.UpperNewMin()) {
      // take state with lower function value
      state = (me.LowerState().Fval() < me.UpperState().Fval()) ? me.LowerState() : me.UpperState();
   } else if (me.LowerNewMin()) {
      state = me.LowerState();
   } else if (me.UpperNewMin()) {
      state = me.UpperState();
   }

   return mstatus;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute a contour for parameters ipar and jpar at the given error level
/// with Minuit2's MnContours, which requires a valid function minimum.

bool RooMinimizer::minuit2Contour(unsigned int ipar, unsigned int jpar, unsigned int npoints, double errorDef,
                                  double *x, double *y)
{
   using namespace ROOT::Minuit2;

   ROOT::Math::MinimizerOptions const &options = _config.MinimizerOptions();
   const int printLevel = options.PrintLevel();
   MnPrint print("RooMinimizer::contour", printLevel);

   if (!_minuit2->minimum) {
      print.Error("No function minimum existing; must minimize function before");
      return false;
   }

   if (!_minuit2->minimum->IsValid()) {
      print.Error("Invalid function minimum");
      return false;
   }

   setMinuit2ErrorDef(errorDef);

   print.Info("Computing contours at level -", errorDef);

   // switch off Minuit2 printing (for level of 0,1)
   Minuit2PrintLevelRAII printLevelRAII{printLevel - 1};

   if (options.Precision() > 0) {
      _minuit2->state.SetPrecision(options.Precision());
   }

   // eventually one should specify tolerance in contours
   MnContours contour(*_fcn, *_minuit2->minimum, makeMinuit2Strategy(options));

   std::vector<std::pair<double, double>> result = contour(ipar, jpar, npoints);
   if (result.size() != npoints) {
      print.Error("Invalid result from MnContours");
      return false;
   }
   for (unsigned int i = 0; i < npoints; ++i) {
      x[i] = result[i].first;
      y[i] = result[i].second;
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the internal fit result from the Minuit2 parameter state.

void RooMinimizer::fillResultFromMinuit2(bool isValid)
{
   using namespace ROOT::Minuit2;

   MnUserParameterState const &state = _minuit2->state;
   std::vector<MinuitParameter> const &params = state.MinuitParameters();
   const unsigned int npar = params.size();

   _result->fValid = isValid;
   _result->fStatus = _minuit2->status;
   _result->fCovStatus = minuit2CovMatrixStatus(_minuit2->minimum.get(), state);
   _result->fVal = state.Fval();
   _result->fEdm = state.Edm();
   _result->fMinimType = _config.MinimizerName();

   _result->fParams.resize(npar);
   _result->fErrors.resize(npar);
   for (unsigned int i = 0; i < npar; ++i) {
      _result->fParams[i] = params[i].Value();
      _result->fErrors[i] = isFixedOrConst(params[i]) ? 0. : params[i].Error();
   }

   // check for fixed parameters
   for (unsigned int ipar = 0; ipar < npar; ++ipar) {
      if (_config.ParSettings(ipar).IsFixed())
         _result->fFixedParams[ipar] = true;
   }

   // fill error matrix in packed lower-triangular storage
   _result->fCovMatrix.clear();
   if (_result->fCovStatus != 0) {
      _result->fCovMatrix.reserve(npar * (npar + 1) / 2);
      for (unsigned int i = 0; i < npar; ++i) {
         for (unsigned int j = 0; j <= i; ++j) {
            _result->fCovMatrix.push_back(minuit2Covariance(state, i, j));
         }
      }
   }

   // global correlation coefficients, zero for fixed parameters
   _result->fGlobalCC.clear();
   MnGlobalCorrelationCoeff const globalCC = state.GlobalCC();
   if (globalCC.IsValid()) {
      _result->fGlobalCC.resize(npar);
      for (unsigned int i = 0; i < npar; ++i) {
         _result->fGlobalCC[i] = isFixedOrConst(params[i]) ? 0. : globalCC.GlobalCC()[state.IntOfExt(i)];
      }
   }
   // minos errors are set separately when calling calculateMinosErrors()
}

double RooMinimizer::FitResult::lowerError(unsigned int i) const
{
   // return lower Minos error for parameter i
   //  return the parabolic error if Minos error has not been calculated for the parameter i
   auto itr = fMinosErrors.find(i);
   return (itr != fMinosErrors.end()) ? itr->second.first : error(i);
}

double RooMinimizer::FitResult::upperError(unsigned int i) const
{
   // return upper Minos error for parameter i
   //  return the parabolic error if Minos error has not been calculated for the parameter i
   auto itr = fMinosErrors.find(i);
   return (itr != fMinosErrors.end()) ? itr->second.second : error(i);
}

bool RooMinimizer::FitResult::isParameterFixed(unsigned int ipar) const
{
   return fFixedParams.find(ipar) != fFixedParams.end();
}

void RooMinimizer::FitResult::GetCovarianceMatrix(TMatrixDSym &covs) const
{
   const size_t nParams = fParams.size();
   covs.ResizeTo(nParams, nParams);
   for (std::size_t ic = 0; ic < nParams; ic++) {
      for (std::size_t ii = 0; ii < nParams; ii++) {
         covs(ic, ii) = covMatrix(fCovMatrix, ic, ii);
      }
   }
}
