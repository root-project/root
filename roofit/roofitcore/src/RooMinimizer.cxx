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

Wrapper class around ROOT::Math::Minimizer that
provides a seamless interface between the minimizer functionality
and the native RooFit interface.
By default the Minimizer is Minuit 2.
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
Various methods are available to control verbosity, profiling,
automatic PDF optimization.
**/

#include "RooMinimizer.h"

#include "RooAbsMinimizerFcn.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsReal.h"
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooSentinel.h"
#include "RooMsgService.h"
#include "RooPlot.h"
#include "RooMinimizerFcn.h"
#include "RooFitResult.h"
#include "TestStatistics/MinuitFcnGrad.h"
#include "RooFit/TestStatistics/RooAbsL.h"
#include "RooFit/TestStatistics/RooRealL.h"
#ifdef ROOFIT_MULTIPROCESS
#include "RooFit/MultiProcess/Config.h"
#include "RooFit/MultiProcess/ProcessTimer.h"
#endif

#include <Fit/BasicFCN.h>
#include <Math/Minimizer.h>
#include <TClass.h>
#include <TGraph.h>
#include <TMarker.h>

#include <fstream>
#include <iostream>
#include <stdexcept> // logic_error


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
RooMinimizer::RooMinimizer(RooAbsReal &function, Config const &cfg) : _cfg(cfg)
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
            "Parallel minimization requested, but ROOT was not compiled with multiprocessing enabled, "
            "please recompile with -Droofit_multiprocess=ON for parallel evaluation");
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
};

/// Initialize the part of the minimizer that is independent of the function to be minimized
void RooMinimizer::initMinimizerFirstPart()
{
   RooSentinel::activate();
   setMinimizerType("");

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
      throw std::logic_error("ProcessTimer, but ROOT was not compiled with multiprocessing enabled, "
                             "please recompile with -Droofit_multiprocess=ON for logging with the "
                             "ProcessTimer.");
#endif
   }
   _fcn->Synchronize(_config.ParamsSettings());

   setMinimizerType(type);
   _config.SetMinimizer(_cfg.minimizerType.c_str(), alg);

   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      bool ret = fitFCN(*_fcn->getMultiGenFcn());
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
         ret = fitFCN(*_fcn->getMultiGenFcn());
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
   if (_minimizer == nullptr) {
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
   if (_minimizer == nullptr) {
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
   if (_minimizer == nullptr) {
      coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!" << std::endl;
      _status = -1;
   } else if (!minosParamList.empty()) {

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
/// If flag is true, perform constant term optimization on
/// function being minimized.

void RooMinimizer::optimizeConst(int flag)
{
   _fcn->setOptimizeConst(flag);
}

////////////////////////////////////////////////////////////////////////////////
/// Save and return a RooFitResult snapshot of current minimizer status.
/// This snapshot contains the values of all constant parameters,
/// the value of all floating parameters at RooMinimizer construction and
/// after the last MINUIT operation, the MINUIT status, variance quality,
/// EDM setting, number of calls with evaluation problems, the minimized
/// function value and the full correlation matrix.

RooFit::OwningPtr<RooFitResult> RooMinimizer::save(const char *userName, const char *userTitle)
{
   if (_minimizer == nullptr) {
      coutW(Minimization) << "RooMinimizer::save: Error, run minimization before!" << std::endl;
      return nullptr;
   }

   TString name = userName ? userName : Form("%s", _fcn->getFunctionName().c_str());
   TString title = userTitle ? userTitle : Form("%s", _fcn->getFunctionTitle().c_str());
   auto fitRes = std::make_unique<RooFitResult>(name, title);

   fitRes->setConstParList(_fcn->constParams());

   fitRes->setNumInvalidNLL(_fcn->GetNumInvalidNLL());

   fitRes->setStatus(_status);
   fitRes->setCovQual(_minimizer->CovMatrixStatus());
   fitRes->setMinNLL(_result->fVal -_fcn->getOffset());
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
   std::vector<double> globalCC;
   TMatrixDSym corrs(nParams);
   TMatrixDSym covs(nParams);
   for (std::size_t ic = 0; ic < nParams; ic++) {
      globalCC.push_back(_result->fGlobalCC[ic]);
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

   // check first if a inimizer is available. If not means
   // the minimization is not done , so do it
   if (_minimizer == nullptr) {
      coutW(Minimization) << "RooMinimizer::contour: Error, run Migrad before contours!" << std::endl;
      return frame;
   }

   // remember our original value of ERRDEF
   double errdef = _minimizer->ErrorDef();

   double n[6];
   n[0] = n1;
   n[1] = n2;
   n[2] = n3;
   n[3] = n4;
   n[4] = n5;
   n[5] = n6;

   for (int ic = 0; ic < 6; ic++) {
      if (n[ic] > 0) {

         // set the value corresponding to an n1-sigma contour
         _minimizer->SetErrorDef(n[ic] * n[ic] * errdef);

         // calculate and draw the contour
         std::vector<double> xcoor(npoints + 1);
         std::vector<double> ycoor(npoints + 1);
         bool ret = _minimizer->Contour(index1, index2, npoints, xcoor.data(), ycoor.data());

         if (!ret) {
            coutE(Minimization) << "RooMinimizer::contour(" << GetName()
                                << ") ERROR: MINUIT did not return a contour graph for n=" << n[ic] << std::endl;
         } else {
            xcoor[npoints] = xcoor[0];
            ycoor[npoints] = ycoor[0];
            TGraph *graph = new TGraph(npoints + 1, xcoor.data(), ycoor.data());

            graph->SetName(Form("contour_%s_n%f", _fcn->getFunctionName().c_str(), n[ic]));
            graph->SetLineStyle(ic + 1);
            graph->SetLineWidth(2);
            graph->SetLineColor(kBlue);
            frame->addObject(graph, "L");
         }
      }
   }

   // restore the original ERRDEF
   _minimizer->SetErrorDef(errdef);

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

ROOT::Math::IMultiGenFunction *RooMinimizer::getMultiGenFcn() const
{
   return _fcn->getMultiGenFcn();
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

   if (_minimizer == nullptr) {
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
   res->setCovQual(_minimizer->CovMatrixStatus());
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

bool RooMinimizer::fitFCN(const ROOT::Math::IMultiGenFunction &fcn)
{
   // fit a user provided FCN function
   // create fit parameter settings
   unsigned int npar = fcn.NDim();
   if (npar == 0) {
      coutE(Minimization) << "RooMinimizer::fitFCN(): FCN function has zero parameters" << std::endl;
      return false;
   }

   // init the minimizer
   initMinimizer();
   // perform the minimization

   // perform the minimization (assume we have already initialized the minimizer)

   bool isValid = _minimizer->Minimize();

   if (!_result)
      _result = std::make_unique<FitResult>();

   fillResult(isValid);

   // set also new parameter values and errors in FitConfig
   if (isValid)
      updateFitConfig();

   return isValid;
}

bool RooMinimizer::calculateHessErrors()
{
   // compute the Hesse errors according to configuration
   // set in the parameters and append value in fit result

   // update  minimizer (recreate if not done or if name has changed
   if (!updateMinimizerOptions()) {
      coutE(Minimization) << "RooMinimizer::calculateHessErrors() Error re-initializing the minimizer" << std::endl;
      return false;
   }

   // run Hesse
   bool ret = _minimizer->Hesse();
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

   // update  minimizer (but cannot re-create in this case). Must use an existing one
   if (!updateMinimizerOptions(false)) {
      coutE(Minimization) << "RooMinimizer::calculateHessErrors() Error re-initializing the minimizer" << std::endl;
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
         bool ret = _minimizer->GetMinosError(index, elow, eup);
         // flags case when a new minimum has been found
         if ((_minimizer->MinosStatus() & 8) != 0) {
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
   _minimizer = std::unique_ptr<ROOT::Math::Minimizer>(_config.CreateMinimizer());
   _minimizer->SetFunction(*getMultiGenFcn());
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

   // update global CC
   _result->fGlobalCC.resize(npar);
   for (unsigned int i = 0; i < npar; ++i) {
      double globcc = min.GlobalCC(i);
      if (globcc < 0) {
         _result->fGlobalCC.clear();
         break; // it is not supported by that minimizer
      }
      _result->fGlobalCC[i] = globcc;
   }
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
