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

RooMinimizer is a wrapper class around ROOT::Fit:Fitter that
provides a seamless interface between the minimizer functionality
and the native RooFit interface.
By default the Minimizer is MINUIT for classic mode and MINUIT2
for parallelized mode (activated with the `RooFit::ModularL(true)`
parameter or by passing a RooRealL function as the minimization
target).
RooMinimizer can minimize any RooAbsReal function with respect to
its parameters. Usual choices for minimization are RooNLLVar
and RooChi2Var
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
#ifdef R__HAS_ROOFIT_MULTIPROCESS
#include "RooFit/MultiProcess/Config.h"
#include "RooFit/MultiProcess/ProcessTimer.h"
#endif

#include "TClass.h"
#include "Math/Minimizer.h"
#include "TMarker.h"
#include "TGraph.h"
#include "Fit/FitConfig.h"

#include <fstream>
#include <iostream>
#include <stdexcept> // logic_error

using namespace std;

ClassImp(RooMinimizer);
;

std::unique_ptr<ROOT::Fit::Fitter> RooMinimizer::_theFitter = {};

////////////////////////////////////////////////////////////////////////////////
/// Cleanup method called by atexit handler installed by RooSentinel
/// to delete all global heap objects when the program is terminated

void RooMinimizer::cleanup()
{
   _theFitter.reset();
}

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
#ifdef R__HAS_ROOFIT_MULTIPROCESS
         if (!_cfg.enableParallelGradient) {
            // Note that this is necessary because there is currently no serial-mode LikelihoodGradientWrapper.
            // We intend to repurpose RooGradMinimizerFcn to build such a LikelihoodGradientSerial class.
            coutI(InputArguments) << "Modular likelihood detected and likelihood parallelization requested, "
                                  << "also setting parallel gradient calculation mode." << std::endl;
            _cfg.enableParallelGradient = 1;
         }
         // If _cfg.parallelize is larger than zero set the number of workers to that value. Otherwise do not do
         // anything and let RooFit::MultiProcess handle the number of workers
         if (_cfg.parallelize > 0)
            RooFit::MultiProcess::Config::setDefaultNWorkers(_cfg.parallelize);
         RooFit::MultiProcess::Config::setTimingAnalysis(_cfg.timingAnalysis);

         _fcn = std::make_unique<RooFit::TestStatistics::MinuitFcnGrad>(
            nll_real->getRooAbsL(), this, _theFitter->Config().ParamsSettings(),
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

   _theFitter = std::make_unique<ROOT::Fit::Fitter>();
   _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str());
   setEps(1.0); // default tolerance
}

/// Initialize the part of the minimizer that is dependent on the function to be minimized
void RooMinimizer::initMinimizerFcnDependentPart(double defaultErrorLevel)
{
   // default max number of calls
   _theFitter->Config().MinimizerOptions().SetMaxIterations(500 * _fcn->getNDim());
   _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(500 * _fcn->getNDim());

   // Shut up for now
   setPrintLevel(-1);

   // Use +0.5 for 1-sigma errors
   setErrorLevel(defaultErrorLevel);

   // Declare our parameters to MINUIT
   _fcn->Synchronize(_theFitter->Config().ParamsSettings());

   // Now set default verbosity
   if (RooMsgService::instance().silentMode()) {
      setPrintLevel(-1);
   } else {
      setPrintLevel(1);
   }

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
   _theFitter->Config().MinimizerOptions().SetStrategy(istrat);
}

////////////////////////////////////////////////////////////////////////////////
/// Change maximum number of MINUIT iterations
/// (RooMinimizer default 500 * #%parameters)

void RooMinimizer::setMaxIterations(int n)
{
   _theFitter->Config().MinimizerOptions().SetMaxIterations(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Change maximum number of likelihood function calss from MINUIT
/// (RooMinimizer default 500 * #%parameters)

void RooMinimizer::setMaxFunctionCalls(int n)
{
   _theFitter->Config().MinimizerOptions().SetMaxFunctionCalls(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the level for MINUIT error analysis to the given
/// value. This function overrides the default value
/// that is taken in the RooMinimizer constructor from
/// the defaultErrorLevel() method of the input function

void RooMinimizer::setErrorLevel(double level)
{
   _theFitter->Config().MinimizerOptions().SetErrorDef(level);
}

////////////////////////////////////////////////////////////////////////////////
/// Change MINUIT epsilon

void RooMinimizer::setEps(double eps)
{
   _theFitter->Config().MinimizerOptions().SetTolerance(eps);
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

////////////////////////////////////////////////////////////////////////////////
/// Return underlying ROOT fitter object

ROOT::Fit::Fitter *RooMinimizer::fitter()
{
   return _theFitter.get();
}

////////////////////////////////////////////////////////////////////////////////
/// Return underlying ROOT fitter object

ROOT::Fit::Fitter const *RooMinimizer::fitter() const
{
   return _theFitter.get();
}

bool RooMinimizer::fitFcn() const
{
   return _fcn->fit(*_theFitter);
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
   if (_cfg.timingAnalysis) 
#ifdef R__HAS_ROOFIT_MULTIPROCESS
      addParamsToProcessTimer();
#else
      throw std::logic_error("ProcessTimer, but ROOT was not compiled with multiprocessing enabled, "
                             "please recompile with -Droofit_multiprocess=ON for logging with the "
                             "ProcessTimer.");
#endif
   _fcn->Synchronize(_theFitter->Config().ParamsSettings());

   setMinimizerType(type);
   _theFitter->Config().SetMinimizer(type, alg);

   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      bool ret = fitFcn();
      _status = ((ret) ? _theFitter->Result().Status() : -1);
   }
   profileStop();
   _fcn->BackProp(_theFitter->Result());

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
   _fcn->Synchronize(_theFitter->Config().ParamsSettings());
   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str(), "migrad");
      bool ret = fitFcn();
      _status = ((ret) ? _theFitter->Result().Status() : -1);
   }
   profileStop();
   _fcn->BackProp(_theFitter->Result());

   saveStatus("MIGRAD", _status);

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute HESSE. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::hesse()
{
   if (_theFitter->GetMinimizer() == 0) {
      coutW(Minimization) << "RooMinimizer::hesse: Error, run Migrad before Hesse!" << endl;
      _status = -1;
   } else {

      _fcn->Synchronize(_theFitter->Config().ParamsSettings());
      profileStart();
      {
         auto ctx = makeEvalErrorContext();

         _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str());
         bool ret = _theFitter->CalculateHessErrors();
         _status = ((ret) ? _theFitter->Result().Status() : -1);
      }
      profileStop();
      _fcn->BackProp(_theFitter->Result());

      saveStatus("HESSE", _status);
   }

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute MINOS. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::minos()
{
   if (_theFitter->GetMinimizer() == 0) {
      coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!" << endl;
      _status = -1;
   } else {

      _fcn->Synchronize(_theFitter->Config().ParamsSettings());
      profileStart();
      {
         auto ctx = makeEvalErrorContext();

         _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str());
         bool ret = _theFitter->CalculateMinosErrors();
         _status = ((ret) ? _theFitter->Result().Status() : -1);
      }

      profileStop();
      _fcn->BackProp(_theFitter->Result());

      saveStatus("MINOS", _status);
   }

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute MINOS for given list of parameters. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::minos(const RooArgSet &minosParamList)
{
   if (_theFitter->GetMinimizer() == 0) {
      coutW(Minimization) << "RooMinimizer::minos: Error, run Migrad before Minos!" << endl;
      _status = -1;
   } else if (!minosParamList.empty()) {

      _fcn->Synchronize(_theFitter->Config().ParamsSettings());
      profileStart();
      {
         auto ctx = makeEvalErrorContext();

         // get list of parameters for Minos
         std::vector<unsigned int> paramInd;
         for (RooAbsArg *arg : minosParamList) {
            RooAbsArg *par = _fcn->GetFloatParamList()->find(arg->GetName());
            if (par && !par->isConstant()) {
               int index = _fcn->GetFloatParamList()->index(par);
               paramInd.push_back(index);
            }
         }

         if (paramInd.size()) {
            // set the parameter indeces
            _theFitter->Config().SetMinosErrors(paramInd);

            _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str());
            bool ret = _theFitter->CalculateMinosErrors();
            _status = ((ret) ? _theFitter->Result().Status() : -1);
            // to avoid that following minimization computes automatically the Minos errors
            _theFitter->Config().SetMinosErrors(false);
         }
      }
      profileStop();
      _fcn->BackProp(_theFitter->Result());

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
   _fcn->Synchronize(_theFitter->Config().ParamsSettings());
   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str(), "seek");
      bool ret = fitFcn();
      _status = ((ret) ? _theFitter->Result().Status() : -1);
   }
   profileStop();
   _fcn->BackProp(_theFitter->Result());

   saveStatus("SEEK", _status);

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute SIMPLEX. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::simplex()
{
   _fcn->Synchronize(_theFitter->Config().ParamsSettings());
   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str(), "simplex");
      bool ret = fitFcn();
      _status = ((ret) ? _theFitter->Result().Status() : -1);
   }
   profileStop();
   _fcn->BackProp(_theFitter->Result());

   saveStatus("SEEK", _status);

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute IMPROVE. Changes in parameter values
/// and calculated errors are automatically
/// propagated back the RooRealVars representing
/// the floating parameters in the MINUIT operation.

int RooMinimizer::improve()
{
   _fcn->Synchronize(_theFitter->Config().ParamsSettings());
   profileStart();
   {
      auto ctx = makeEvalErrorContext();

      _theFitter->Config().SetMinimizer(_cfg.minimizerType.c_str(), "migradimproved");
      bool ret = fitFcn();
      _status = ((ret) ? _theFitter->Result().Status() : -1);
   }
   profileStop();
   _fcn->BackProp(_theFitter->Result());

   saveStatus("IMPROVE", _status);

   return _status;
}

////////////////////////////////////////////////////////////////////////////////
/// Change the MINUIT internal printing level

void RooMinimizer::setPrintLevel(int newLevel)
{
   _theFitter->Config().MinimizerOptions().SetPrintLevel(newLevel + 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Get the MINUIT internal printing level

int RooMinimizer::getPrintLevel()
{
   return _theFitter->Config().MinimizerOptions().PrintLevel() + 1;
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

RooFitResult *RooMinimizer::save(const char *userName, const char *userTitle)
{
   if (_theFitter->GetMinimizer() == 0) {
      coutW(Minimization) << "RooMinimizer::save: Error, run minimization before!" << endl;
      return nullptr;
   }

   TString name, title;
   name = userName ? userName : Form("%s", _fcn->getFunctionName().c_str());
   title = userTitle ? userTitle : Form("%s", _fcn->getFunctionTitle().c_str());
   RooFitResult *fitRes = new RooFitResult(name, title);

   // Move eventual fixed parameters in floatList to constList
   RooArgList saveConstList(*(_fcn->GetConstParamList()));
   RooArgList saveFloatInitList(*(_fcn->GetInitFloatParamList()));
   RooArgList saveFloatFinalList(*(_fcn->GetFloatParamList()));
   for (std::size_t i = 0; i < _fcn->GetFloatParamList()->size(); i++) {
      RooAbsArg *par = _fcn->GetFloatParamList()->at(i);
      if (par->isConstant()) {
         saveFloatInitList.remove(*saveFloatInitList.find(par->GetName()), true);
         saveFloatFinalList.remove(*par);
         saveConstList.add(*par);
      }
   }
   saveConstList.sort();

   fitRes->setConstParList(saveConstList);
   fitRes->setInitParList(saveFloatInitList);

   // The fitter often clones the function. We therefore have to ask it for its copy.
   const auto fitFcn = dynamic_cast<const RooAbsMinimizerFcn *>(_theFitter->GetFCN());
   double removeOffset = 0.;
   if (fitFcn) {
      fitRes->setNumInvalidNLL(fitFcn->GetNumInvalidNLL());
      removeOffset = -fitFcn->getOffset();
   }

   fitRes->setStatus(_status);
   fitRes->setCovQual(_theFitter->GetMinimizer()->CovMatrixStatus());
   fitRes->setMinNLL(_theFitter->Result().MinFcnValue() + removeOffset);
   fitRes->setEDM(_theFitter->Result().Edm());
   fitRes->setFinalParList(saveFloatFinalList);
   if (!_extV) {
      std::vector<double> globalCC;
      TMatrixDSym corrs(_theFitter->Result().Parameters().size());
      TMatrixDSym covs(_theFitter->Result().Parameters().size());
      for (std::size_t ic = 0; ic < _theFitter->Result().Parameters().size(); ic++) {
         globalCC.push_back(_theFitter->Result().GlobalCC(ic));
         for (std::size_t ii = 0; ii < _theFitter->Result().Parameters().size(); ii++) {
            corrs(ic, ii) = _theFitter->Result().Correlation(ic, ii);
            covs(ic, ii) = _theFitter->Result().CovMatrix(ic, ii);
         }
      }
      fitRes->fillCorrMatrix(globalCC, corrs, covs);
   } else {
      fitRes->setCovarianceMatrix(*_extV);
   }

   fitRes->setStatusHistory(_statusHistory);

   return fitRes;
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
   RooArgList *params = _fcn->GetFloatParamList();
   std::unique_ptr<RooArgList> paramSave{static_cast<RooArgList *>(params->snapshot())};

   // Verify that both variables are floating parameters of PDF
   int index1 = _fcn->GetFloatParamList()->index(&var1);
   if (index1 < 0) {
      coutE(Minimization) << "RooMinimizer::contour(" << GetName() << ") ERROR: " << var1.GetName()
                          << " is not a floating parameter of " << _fcn->getFunctionName() << endl;
      return nullptr;
   }

   int index2 = _fcn->GetFloatParamList()->index(&var2);
   if (index2 < 0) {
      coutE(Minimization) << "RooMinimizer::contour(" << GetName() << ") ERROR: " << var2.GetName()
                          << " is not a floating parameter of PDF " << _fcn->getFunctionName() << endl;
      return nullptr;
   }

   // create and draw a frame
   RooPlot *frame = new RooPlot(var1, var2);

   // draw a point at the current parameter values
   TMarker *point = new TMarker(var1.getVal(), var2.getVal(), 8);
   frame->addObject(point);

   // check first if a inimizer is available. If not means
   // the minimization is not done , so do it
   if (_theFitter->GetMinimizer() == 0) {
      coutW(Minimization) << "RooMinimizer::contour: Error, run Migrad before contours!" << endl;
      return frame;
   }

   // remember our original value of ERRDEF
   double errdef = _theFitter->GetMinimizer()->ErrorDef();

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
         _theFitter->GetMinimizer()->SetErrorDef(n[ic] * n[ic] * errdef);

         // calculate and draw the contour
         std::vector<double> xcoor(npoints + 1);
         std::vector<double> ycoor(npoints + 1);
         bool ret = _theFitter->GetMinimizer()->Contour(index1, index2, npoints, xcoor.data(), ycoor.data());

         if (!ret) {
            coutE(Minimization) << "RooMinimizer::contour(" << GetName()
                                << ") ERROR: MINUIT did not return a contour graph for n=" << n[ic] << endl;
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
   _theFitter->GetMinimizer()->SetErrorDef(errdef);

   // restore parameter values
   params->assign(*paramSave);

   return frame;
}

////////////////////////////////////////////////////////////////////////////////
/// Add parameters in metadata field to process timer

void RooMinimizer::addParamsToProcessTimer()
{
#ifdef R__HAS_ROOFIT_MULTIPROCESS
   // parameter indices for use in timing heat matrix
   std::vector<std::string> parameter_names;
   for (auto &&parameter : *_fcn->GetFloatParamList()) {
      parameter_names.push_back(parameter->GetName());
      if (_cfg.verbose) {
         coutI(Minimization) << "parameter name: " << parameter_names.back() << std::endl;
      }
   }
   RooFit::MultiProcess::ProcessTimer::add_metadata(parameter_names);
#else
   coutI(Minimization) << "Not adding parameters to processtimer because multiprocessing "
                       << "is not enabled." << std::endl;
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
   auto *fitterFcn = fitter()->GetFCN();
   return fitterFcn ? fitterFcn : _fcn->getMultiGenFcn();
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

RooFitResult *RooMinimizer::lastMinuitFit()
{
   return RooMinimizer::lastMinuitFit({});
}

RooFitResult *RooMinimizer::lastMinuitFit(const RooArgList &varList)
{
   // Import the results of the last fit performed, interpreting
   // the fit parameters as the given varList of parameters.

   if (_theFitter == 0 || _theFitter->GetMinimizer() == 0) {
      oocoutE(nullptr, InputArguments) << "RooMinimizer::save: Error, run minimization before!" << endl;
      return nullptr;
   }

   // Verify length of supplied varList
   if (!varList.empty() && varList.size() != _theFitter->Result().NTotalParameters()) {
      oocoutE(nullptr, InputArguments)
         << "RooMinimizer::lastMinuitFit: ERROR: supplied variable list must be either empty " << endl
         << "                             or match the number of variables of the last fit ("
         << _theFitter->Result().NTotalParameters() << ")" << endl;
      return nullptr;
   }

   // Verify that all members of varList are of type RooRealVar
   for (RooAbsArg *arg : varList) {
      if (!dynamic_cast<RooRealVar *>(arg)) {
         oocoutE(nullptr, InputArguments) << "RooMinimizer::lastMinuitFit: ERROR: variable '" << arg->GetName()
                                          << "' is not of type RooRealVar" << endl;
         return nullptr;
      }
   }

   RooFitResult *res = new RooFitResult("lastMinuitFit", "Last MINUIT fit");

   // Extract names of fit parameters
   // and construct corresponding RooRealVars
   RooArgList constPars("constPars");
   RooArgList floatPars("floatPars");

   unsigned int i;
   for (i = 0; i < _theFitter->Result().NTotalParameters(); ++i) {

      TString varName(_theFitter->Result().GetParameterName(i));
      bool isConst(_theFitter->Result().IsParameterFixed(i));

      double xlo = _theFitter->Config().ParSettings(i).LowerLimit();
      double xhi = _theFitter->Config().ParSettings(i).UpperLimit();
      double xerr = _theFitter->Result().Error(i);
      double xval = _theFitter->Result().Value(i);

      std::unique_ptr<RooRealVar> var;
      if (varList.empty()) {

         if ((xlo < xhi) && !isConst) {
            var = std::make_unique<RooRealVar>(varName, varName, xval, xlo, xhi);
         } else {
            var = std::make_unique<RooRealVar>(varName, varName, xval);
         }
         var->setConstant(isConst);
      } else {

         var.reset(static_cast<RooRealVar *>(varList.at(i)->Clone()));
         var->setConstant(isConst);
         var->setVal(xval);
         if (xlo < xhi) {
            var->setRange(xlo, xhi);
         }

         if (varName.CompareTo(var->GetName())) {
            oocoutI(nullptr, Eval) << "RooMinimizer::lastMinuitFit: fit parameter '" << varName
                                   << "' stored in variable '" << var->GetName() << "'" << endl;
         }
      }

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
   res->setMinNLL(_theFitter->Result().MinFcnValue());
   res->setEDM(_theFitter->Result().Edm());
   res->setCovQual(_theFitter->GetMinimizer()->CovMatrixStatus());
   res->setStatus(_theFitter->Result().Status());
   std::vector<double> globalCC;
   TMatrixDSym corrs(_theFitter->Result().Parameters().size());
   TMatrixDSym covs(_theFitter->Result().Parameters().size());
   for (unsigned int ic = 0; ic < _theFitter->Result().Parameters().size(); ic++) {
      globalCC.push_back(_theFitter->Result().GlobalCC(ic));
      for (unsigned int ii = 0; ii < _theFitter->Result().Parameters().size(); ii++) {
         corrs(ic, ii) = _theFitter->Result().Correlation(ic, ii);
         covs(ic, ii) = _theFitter->Result().CovMatrix(ic, ii);
      }
   }
   res->fillCorrMatrix(globalCC, corrs, covs);

   return res;
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
   if (_cfg.logf)
      return _fcn->SetLogFile(_cfg.logf);
   else
      return false;
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

int RooMinimizer::Config::getDefaultWorkers()
{
#ifdef R__HAS_ROOFIT_MULTIPROCESS
   return RooFit::MultiProcess::Config::getDefaultNWorkers();
#else
   return 0;
#endif
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
