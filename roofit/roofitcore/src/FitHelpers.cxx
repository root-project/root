/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "FitHelpers.h"

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooAddition.h>
#include <RooBatchCompute.h>
#include <RooBinSamplingPdf.h>
#include <RooCategory.h>
#include <RooCmdConfig.h>
#include <RooConstraintSum.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooDerivative.h>
#include <RooFit/Evaluator.h>
#include <RooFit/TestStatistics/RooRealL.h>
#include <RooFit/TestStatistics/buildLikelihood.h>
#include <RooFitResult.h>
#include <RooLinkedList.h>
#include <RooMinimizer.h>
#include <RooConstVar.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooFormulaVar.h>

#include <Math/CholeskyDecomp.h>
#include <Math/Util.h>

#include "ConstraintHelpers.h"
#include "RooEvaluatorWrapper.h"
#include "RooFitImplHelpers.h"
#include "RooFit/Detail/RooNLLVarNew.h"

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
#include "RooChi2Var.h"
#include "RooNLLVar.h"
#include "RooXYChi2Var.h"
#endif

using RooFit::Detail::RooNLLVarNew;

namespace {

constexpr int extendedFitDefault = 2;

////////////////////////////////////////////////////////////////////////////////
/// Use the asymptotically correct approach to estimate errors in the presence of weights.
/// This is slower but more accurate than `SumW2Error`. See also https://arxiv.org/abs/1911.01303).
/// Applies the calculated covaraince matrix to the RooMinimizer and returns
/// the quality of the covariance matrix.
/// See also the documentation of RooAbsPdf::fitTo(), where this function is used.
/// \param[in] minimizer The RooMinimizer to get the fit result from. The state
///            of the minimizer will be altered by this function: the covariance
///            matrix caltulated here will be applied to it via
///            RooMinimizer::applyCovarianceMatrix().
/// \param[in] data The dataset that was used for the fit.
int calcAsymptoticCorrectedCovariance(RooAbsReal &pdf, RooMinimizer &minimizer, RooAbsData const &data)
{
   RooFormulaVar logpdf("logpdf", "log(pdf)", "log(@0)", pdf);
   RooArgSet obs;
   logpdf.getObservables(data.get(), obs);

   // Warning if the dataset is binned. TODO: in some cases,
   // people also use RooDataSet to encode binned data,
   // e.g. for simultaneous fits. It would be useful to detect
   // this in this future as well.
   if (dynamic_cast<RooDataHist const *>(&data)) {
      oocoutW(&pdf, InputArguments)
         << "RooAbsPdf::fitTo(" << pdf.GetName()
         << ") WARNING: Asymptotic error correction is requested for a binned data set. "
            "This method is not designed to handle binned data. A standard chi2 fit will likely be more suitable.";
   };

   // Calculated corrected errors for weighted likelihood fits
   std::unique_ptr<RooFitResult> rw(minimizer.save());
   // Weighted inverse Hessian matrix
   const TMatrixDSym &matV = rw->covarianceMatrix();
   oocoutI(&pdf, Fitting)
      << "RooAbsPdf::fitTo(" << pdf.GetName()
      << ") Calculating covariance matrix according to the asymptotically correct approach. If you find this "
         "method useful please consider citing https://arxiv.org/abs/1911.01303.\n";

   // Initialise matrix containing first derivatives
   int nFloatPars = rw->floatParsFinal().size();
   TMatrixDSym num(nFloatPars);
   for (int k = 0; k < nFloatPars; k++) {
      for (int l = 0; l < nFloatPars; l++) {
         num(k, l) = 0.0;
      }
   }

   // Create derivative objects
   std::vector<std::unique_ptr<RooDerivative>> derivatives;
   const RooArgList &floated = rw->floatParsFinal();
   RooArgSet allparams;
   logpdf.getParameters(data.get(), allparams);
   std::unique_ptr<RooArgSet> floatingparams{allparams.selectByAttrib("Constant", false)};

   const double eps = 1.0e-4;

   // Calculate derivatives of logpdf
   for (const auto paramresult : floated) {
      auto paraminternal = static_cast<RooRealVar *>(floatingparams->find(*paramresult));
      assert(floatingparams->find(*paramresult)->IsA() == RooRealVar::Class());
      double error = static_cast<RooRealVar *>(paramresult)->getError();
      derivatives.emplace_back(logpdf.derivative(*paraminternal, obs, 1, eps * error));
   }

   // Calculate derivatives for number of expected events, needed for extended ML fit
   RooAbsPdf *extended_pdf = dynamic_cast<RooAbsPdf *>(&pdf);
   std::vector<double> diffs_expected(floated.size(), 0.0);
   if (extended_pdf && extended_pdf->expectedEvents(obs) != 0.0) {
      for (std::size_t k = 0; k < floated.size(); k++) {
         const auto paramresult = static_cast<RooRealVar *>(floated.at(k));
         auto paraminternal = static_cast<RooRealVar *>(floatingparams->find(*paramresult));

         *paraminternal = paramresult->getVal();
         double error = paramresult->getError();
         paraminternal->setVal(paramresult->getVal() + eps * error);
         double expected_plus = log(extended_pdf->expectedEvents(obs));
         paraminternal->setVal(paramresult->getVal() - eps * error);
         double expected_minus = log(extended_pdf->expectedEvents(obs));
         *paraminternal = paramresult->getVal();
         double diff = (expected_plus - expected_minus) / (2.0 * eps * error);
         diffs_expected[k] = diff;
      }
   }

   // Loop over data
   for (int j = 0; j < data.numEntries(); j++) {
      // Sets obs to current data point, this is where the pdf will be evaluated
      obs.assign(*data.get(j));
      // Determine first derivatives
      std::vector<double> diffs(floated.size(), 0.0);
      for (std::size_t k = 0; k < floated.size(); k++) {
         const auto paramresult = static_cast<RooRealVar *>(floated.at(k));
         auto paraminternal = static_cast<RooRealVar *>(floatingparams->find(*paramresult));
         // first derivative to parameter k at best estimate point for this measurement
         double diff = derivatives[k]->getVal();
         // need to reset to best fit point after differentiation
         *paraminternal = paramresult->getVal();
         diffs[k] = diff;
      }

      // Fill numerator matrix
      for (std::size_t k = 0; k < floated.size(); k++) {
         for (std::size_t l = 0; l < floated.size(); l++) {
            num(k, l) += data.weightSquared() * (diffs[k] + diffs_expected[k]) * (diffs[l] + diffs_expected[l]);
         }
      }
   }
   num.Similarity(matV);

   // Propagate corrected errors to parameters objects
   minimizer.applyCovarianceMatrix(num);

   // The derivatives are found in RooFit and not with the minimizer (e.g.
   // minuit), so the quality of the corrected covariance matrix corresponds to
   // the quality of the original covariance matrix
   return rw->covQual();
}

////////////////////////////////////////////////////////////////////////////////
/// Apply correction to errors and covariance matrix. This uses two covariance
/// matrices, one with the weights, the other with squared weights, to obtain
/// the correct errors for weighted likelihood fits.
/// Applies the calculated covaraince matrix to the RooMinimizer and returns
/// the quality of the covariance matrix.
/// See also the documentation of RooAbsPdf::fitTo(), where this function is used.
/// \param[in] minimizer The RooMinimizer to get the fit result from. The state
///            of the minimizer will be altered by this function: the covariance
///            matrix caltulated here will be applied to it via
///            RooMinimizer::applyCovarianceMatrix().
/// \param[in] nll The NLL object that was used for the fit.
int calcSumW2CorrectedCovariance(RooAbsReal const &pdf, RooMinimizer &minimizer, RooAbsReal &nll)
{
   // Calculated corrected errors for weighted likelihood fits
   std::unique_ptr<RooFitResult> rw{minimizer.save()};
   nll.applyWeightSquared(true);
   oocoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                          << ") Calculating sum-of-weights-squared correction matrix for covariance matrix\n";
   minimizer.hesse();
   std::unique_ptr<RooFitResult> rw2{minimizer.save()};
   nll.applyWeightSquared(false);

   // Apply correction matrix
   const TMatrixDSym &matV = rw->covarianceMatrix();
   TMatrixDSym matC = rw2->covarianceMatrix();
   ROOT::Math::CholeskyDecompGenDim<double> decomp(matC.GetNrows(), matC);
   if (!decomp) {
      oocoutE(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                             << ") ERROR: Cannot apply sum-of-weights correction to covariance matrix: correction "
                                "matrix calculated with weight-squared is singular\n";
      return -1;
   }

   // replace C by its inverse
   decomp.Invert(matC);
   // the class lies about the matrix being symmetric, so fill in the
   // part above the diagonal
   for (int i = 0; i < matC.GetNrows(); ++i) {
      for (int j = 0; j < i; ++j) {
         matC(j, i) = matC(i, j);
      }
   }
   matC.Similarity(matV);
   // C now contains V C^-1 V
   // Propagate corrected errors to parameters objects
   minimizer.applyCovarianceMatrix(matC);

   return std::min(rw->covQual(), rw2->covQual());
}

/// Configuration struct for RooAbsPdf::minimizeNLL with all the default values
/// that also should be taken as the default values for RooAbsPdf::fitTo.
struct MinimizerConfig {
   double recoverFromNaN = 10.;
   int optConst = 2;
   int verbose = 0;
   int doSave = 0;
   int doTimer = 0;
   int printLevel = 1;
   int strategy = 1;
   int initHesse = 0;
   int hesse = 1;
   int minos = 0;
   int numee = 10;
   int doEEWall = 1;
   int doWarn = 1;
   int doSumW2 = -1;
   int doAsymptotic = -1;
   int maxCalls = -1;
   int doOffset = -1;
   int parallelize = 0;
   bool enableParallelGradient = true;
   bool enableParallelDescent = false;
   bool timingAnalysis = false;
   const RooArgSet *minosSet = nullptr;
   std::string minType;
   std::string minAlg = "minuit";
};

bool interpretExtendedCmdArg(RooAbsPdf const &pdf, int extendedCmdArg)
{
   // Process automatic extended option
   if (extendedCmdArg == extendedFitDefault) {
      bool ext = pdf.extendMode() == RooAbsPdf::CanBeExtended || pdf.extendMode() == RooAbsPdf::MustBeExtended;
      if (ext) {
         oocoutI(&pdf, Minimization)
            << "p.d.f. provides expected number of events, including extended term in likelihood." << std::endl;
      }
      return ext;
   }
   // If Extended(false) was explicitly set, but the pdf MUST be extended, then
   // it's time to print an error. This happens when you're fitting a RooAddPdf
   // with coefficient that represent yields, and without the additional
   // constraint these coefficients are degenerate because the RooAddPdf
   // normalizes itself. Nothing correct can come out of this.
   if (extendedCmdArg == 0) {
      if (pdf.extendMode() == RooAbsPdf::MustBeExtended) {
         std::string errMsg = "You used the Extended(false) option on a pdf where the fit MUST be extended! "
                              "The parameters are not well defined and you're getting nonsensical results.";
         oocoutE(&pdf, InputArguments) << errMsg << std::endl;
      }
   }
   return extendedCmdArg;
}

/// To set the fitrange attribute of the PDF and custom ranges for the
/// observables so that RooPlot can automatically plot the fitting range.
void resetFitrangeAttributes(RooAbsArg &pdf, RooAbsData const &data, std::string const &baseName, const char *rangeName,
                             bool splitRange)
{
   // Clear possible range attributes from previous fits.
   pdf.removeStringAttribute("fitrange");

   // No fitrange was specified, so we do nothing. Or "SplitRange" is used, and
   // then there are no uniquely defined ranges for the observables (as they
   // are different in each category).
   if (!rangeName || splitRange)
      return;

   RooArgSet observables;
   pdf.getObservables(data.get(), observables);

   std::string fitrangeValue;
   auto subranges = ROOT::Split(rangeName, ",");
   for (auto const &subrange : subranges) {
      if (subrange.empty())
         continue;
      std::string fitrangeValueSubrange = std::string("fit_") + baseName;
      if (subranges.size() > 1) {
         fitrangeValueSubrange += "_" + subrange;
      }
      fitrangeValue += fitrangeValueSubrange + ",";
      for (RooAbsArg *arg : observables) {

         if (arg->isCategory())
            continue;
         auto &observable = static_cast<RooRealVar &>(*arg);

         observable.setRange(fitrangeValueSubrange.c_str(), observable.getMin(subrange.c_str()),
                             observable.getMax(subrange.c_str()));
      }
   }
   pdf.setStringAttribute("fitrange", fitrangeValue.substr(0, fitrangeValue.size() - 1).c_str());
}

std::unique_ptr<RooAbsArg> createSimultaneousNLL(RooSimultaneous const &simPdf, bool isSimPdfExtended,
                                                 std::string const &rangeName, RooFit::OffsetMode offset)
{
   RooAbsCategoryLValue const &simCat = simPdf.indexCat();

   // Prepare the NLL terms for each component
   RooArgList nllTerms;
   for (auto const &catState : simCat) {
      std::string const &catName = catState.first;
      RooAbsCategory::value_type catIndex = catState.second;

      // If the channel is not in the selected range of the category variable, we
      // won't create an for NLL this channel.
      if (!rangeName.empty()) {
         // Only the RooCategory supports ranges, not the other
         // RooAbsCategoryLValue-derived classes.
         auto simCatAsRooCategory = dynamic_cast<RooCategory const *>(&simCat);
         if (simCatAsRooCategory && !simCatAsRooCategory->isStateInRange(rangeName.c_str(), catIndex)) {
            continue;
         }
      }

      if (RooAbsPdf *pdf = simPdf.getPdf(catName.c_str())) {
         auto name = std::string("nll_") + pdf->GetName();
         std::unique_ptr<RooArgSet> observables{
            std::unique_ptr<RooArgSet>(pdf->getVariables())->selectByAttrib("__obs__", true)};
         // In a simultaneous fit, it is allowed that only a subset of the pdfs
         // are extended. Therefore, we have to make sure that we don't request
         // extended NLL objects for channels that can't be extended.
         const bool isPdfExtended = isSimPdfExtended && pdf->extendMode() != RooAbsPdf::CanNotBeExtended;
         auto nll =
            std::make_unique<RooNLLVarNew>(name.c_str(), name.c_str(), *pdf, *observables, isPdfExtended, offset);
         // Rename the special variables
         nll->setPrefix(std::string("_") + catName + "_");
         nllTerms.addOwned(std::move(nll));
      }
   }

   for (auto *nll : static_range_cast<RooNLLVarNew *>(nllTerms)) {
      nll->setSimCount(nllTerms.size());
   }

   // Time to sum the NLLs
   auto nll = std::make_unique<RooAddition>("mynll", "mynll", nllTerms);
   nll->addOwnedComponents(std::move(nllTerms));
   return nll;
}

std::unique_ptr<RooAbsReal> createNLLNew(RooAbsPdf &pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
                                         std::string const &rangeName, RooArgSet const &projDeps, bool isExtended,
                                         double integrateOverBinsPrecision, RooFit::OffsetMode offset)
{
   if (constraints) {
      // The computation graph for the constraints is very small, no need to do
      // the tracking of clean and dirty nodes here.
      constraints->setOperMode(RooAbsArg::ADirty);
   }

   RooArgSet observables;
   pdf.getObservables(data.get(), observables);
   observables.remove(projDeps, true, true);

   oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                            << ") fixing normalization set for coefficient determination to observables in data"
                            << "\n";
   pdf.fixAddCoefNormalization(observables, false);

   // Deal with the IntegrateBins argument
   RooArgList binSamplingPdfs;
   std::unique_ptr<RooAbsPdf> wrappedPdf = RooBinSamplingPdf::create(pdf, data, integrateOverBinsPrecision);
   RooAbsPdf &finalPdf = wrappedPdf ? *wrappedPdf : pdf;
   if (wrappedPdf) {
      binSamplingPdfs.addOwned(std::move(wrappedPdf));
   }
   // Done dealing with the IntegrateBins option

   RooArgList nllTerms;

   auto simPdf = dynamic_cast<RooSimultaneous *>(&finalPdf);
   if (simPdf) {
      simPdf->wrapPdfsInBinSamplingPdfs(data, integrateOverBinsPrecision);
      nllTerms.addOwned(createSimultaneousNLL(*simPdf, isExtended, rangeName, offset));
   } else {
      nllTerms.addOwned(
         std::make_unique<RooNLLVarNew>("RooNLLVarNew", "RooNLLVarNew", finalPdf, observables, isExtended, offset));
   }
   if (constraints) {
      nllTerms.addOwned(std::move(constraints));
   }

   std::string nllName = std::string("nll_") + pdf.GetName() + "_" + data.GetName();
   auto nll = std::make_unique<RooAddition>(nllName.c_str(), nllName.c_str(), nllTerms);
   nll->addOwnedComponents(std::move(binSamplingPdfs));
   nll->addOwnedComponents(std::move(nllTerms));

   return nll;
}

} // namespace

namespace RooFit {
namespace FitHelpers {

void defineMinimizationOptions(RooCmdConfig &pc)
{
   // Default-initialized instance of MinimizerConfig to get the default
   // minimizer parameter values.
   MinimizerConfig minimizerDefaults;

   pc.defineDouble("RecoverFromUndefinedRegions", "RecoverFromUndefinedRegions", 0, minimizerDefaults.recoverFromNaN);
   pc.defineInt("optConst", "Optimize", 0, minimizerDefaults.optConst);
   pc.defineInt("verbose", "Verbose", 0, minimizerDefaults.verbose);
   pc.defineInt("doSave", "Save", 0, minimizerDefaults.doSave);
   pc.defineInt("doTimer", "Timer", 0, minimizerDefaults.doTimer);
   pc.defineInt("printLevel", "PrintLevel", 0, minimizerDefaults.printLevel);
   pc.defineInt("strategy", "Strategy", 0, minimizerDefaults.strategy);
   pc.defineInt("initHesse", "InitialHesse", 0, minimizerDefaults.initHesse);
   pc.defineInt("hesse", "Hesse", 0, minimizerDefaults.hesse);
   pc.defineInt("minos", "Minos", 0, minimizerDefaults.minos);
   pc.defineInt("numee", "PrintEvalErrors", 0, minimizerDefaults.numee);
   pc.defineInt("doEEWall", "EvalErrorWall", 0, minimizerDefaults.doEEWall);
   pc.defineInt("doWarn", "Warnings", 0, minimizerDefaults.doWarn);
   pc.defineInt("doSumW2", "SumW2Error", 0, minimizerDefaults.doSumW2);
   pc.defineInt("doAsymptoticError", "AsymptoticError", 0, minimizerDefaults.doAsymptotic);
   pc.defineInt("maxCalls", "MaxCalls", 0, minimizerDefaults.maxCalls);
   pc.defineInt("doOffset", "OffsetLikelihood", 0, 0);
   pc.defineInt("parallelize", "Parallelize", 0, 0); // Three parallelize arguments
   pc.defineInt("enableParallelGradient", "ParallelGradientOptions", 0, 0);
   pc.defineInt("enableParallelDescent", "ParallelDescentOptions", 0, 0);
   pc.defineInt("timingAnalysis", "TimingAnalysis", 0, 0);
   pc.defineString("mintype", "Minimizer", 0, minimizerDefaults.minType.c_str());
   pc.defineString("minalg", "Minimizer", 1, minimizerDefaults.minAlg.c_str());
   pc.defineSet("minosSet", "Minos", 0, minimizerDefaults.minosSet);
}

////////////////////////////////////////////////////////////////////////////////
/// Minimizes a given NLL variable by finding the optimal parameters with the
/// RooMinimzer. The NLL variable can be created with RooAbsPdf::createNLL.
/// If you are looking for a function that combines likelihood creation with
/// fitting, see RooAbsPdf::fitTo.
/// \param[in] nll The negative log-likelihood variable to minimize.
/// \param[in] data The dataset that was also used for the NLL. It's a necessary
///            parameter because it is used in the asymptotic error correction.
/// \param[in] cfg Configuration struct with all the configuration options for
///            the RooMinimizer. These are a subset of the options that you can
///            also pass to RooAbsPdf::fitTo via the RooFit command arguments.
std::unique_ptr<RooFitResult> minimize(RooAbsReal &pdf, RooAbsReal &nll, RooAbsData const &data, RooCmdConfig const &pc)
{
   MinimizerConfig cfg;
   cfg.recoverFromNaN = pc.getDouble("RecoverFromUndefinedRegions");
   cfg.optConst = pc.getInt("optConst");
   cfg.verbose = pc.getInt("verbose");
   cfg.doSave = pc.getInt("doSave");
   cfg.doTimer = pc.getInt("doTimer");
   cfg.printLevel = pc.getInt("printLevel");
   cfg.strategy = pc.getInt("strategy");
   cfg.initHesse = pc.getInt("initHesse");
   cfg.hesse = pc.getInt("hesse");
   cfg.minos = pc.getInt("minos");
   cfg.numee = pc.getInt("numee");
   cfg.doEEWall = pc.getInt("doEEWall");
   cfg.doWarn = pc.getInt("doWarn");
   cfg.doSumW2 = pc.getInt("doSumW2");
   cfg.doAsymptotic = pc.getInt("doAsymptoticError");
   cfg.maxCalls = pc.getInt("maxCalls");
   cfg.minosSet = pc.getSet("minosSet");
   cfg.minType = pc.getString("mintype", "");
   cfg.minAlg = pc.getString("minalg", "minuit");
   cfg.doOffset = pc.getInt("doOffset");
   cfg.parallelize = pc.getInt("parallelize");
   cfg.enableParallelGradient = pc.getInt("enableParallelGradient");
   cfg.enableParallelDescent = pc.getInt("enableParallelDescent");
   cfg.timingAnalysis = pc.getInt("timingAnalysis");

   // Determine if the dataset has weights
   bool weightedData = data.isNonPoissonWeighted();

   std::string msgPrefix = std::string{"RooAbsPdf::fitTo("} + pdf.GetName() + "): ";

   // Warn user that a method to determine parameter uncertainties should be provided if weighted data is offered
   if (weightedData && cfg.doSumW2 == -1 && cfg.doAsymptotic == -1) {
      oocoutW(&pdf, InputArguments) << msgPrefix <<
         R"(WARNING: a likelihood fit is requested of what appears to be weighted data.
       While the estimated values of the parameters will always be calculated taking the weights into account,
       there are multiple ways to estimate the errors of the parameters. You are advised to make an
       explicit choice for the error calculation:
           - Either provide SumW2Error(true), to calculate a sum-of-weights-corrected HESSE error matrix
             (error will be proportional to the number of events in MC).
           - Or provide SumW2Error(false), to return errors from original HESSE error matrix
             (which will be proportional to the sum of the weights, i.e., a dataset with <sum of weights> events).
           - Or provide AsymptoticError(true), to use the asymptotically correct expression
             (for details see https://arxiv.org/abs/1911.01303)."
)";
   }

   if (cfg.minos && (cfg.doSumW2 == 1 || cfg.doAsymptotic == 1)) {
      oocoutE(&pdf, InputArguments)
         << msgPrefix
         << " sum-of-weights and asymptotic error correction do not work with MINOS errors. Not fitting.\n";
      return nullptr;
   }
   if (cfg.doAsymptotic == 1 && cfg.minos) {
      oocoutW(&pdf, InputArguments) << msgPrefix << "WARNING: asymptotic correction does not apply to MINOS errors\n";
   }

   // avoid setting both SumW2 and Asymptotic for uncertainty correction
   if (cfg.doSumW2 == 1 && cfg.doAsymptotic == 1) {
      oocoutE(&pdf, InputArguments) << msgPrefix
                                    << "ERROR: Cannot compute both asymptotically correct and SumW2 errors.\n";
      return nullptr;
   }

   // Instantiate RooMinimizer
   RooMinimizer::Config minimizerConfig;
   minimizerConfig.enableParallelGradient = cfg.enableParallelGradient;
   minimizerConfig.enableParallelDescent = cfg.enableParallelDescent;
   minimizerConfig.parallelize = cfg.parallelize;
   minimizerConfig.timingAnalysis = cfg.timingAnalysis;
   minimizerConfig.offsetting = cfg.doOffset;
   RooMinimizer m(nll, minimizerConfig);

   m.setMinimizerType(cfg.minType);
   m.setEvalErrorWall(cfg.doEEWall);
   m.setRecoverFromNaNStrength(cfg.recoverFromNaN);
   m.setPrintEvalErrors(cfg.numee);
   if (cfg.maxCalls > 0)
      m.setMaxFunctionCalls(cfg.maxCalls);
   if (cfg.printLevel != 1)
      m.setPrintLevel(cfg.printLevel);
   if (cfg.optConst)
      m.optimizeConst(cfg.optConst); // Activate constant term optimization
   if (cfg.verbose)
      m.setVerbose(true); // Activate verbose options
   if (cfg.doTimer)
      m.setProfile(true); // Activate timer options
   if (cfg.strategy != 1)
      m.setStrategy(cfg.strategy); // Modify fit strategy
   if (cfg.initHesse)
      m.hesse();                                        // Initialize errors with hesse
   m.minimize(cfg.minType.c_str(), cfg.minAlg.c_str()); // Minimize using chosen algorithm
   if (cfg.hesse)
      m.hesse(); // Evaluate errors with Hesse

   int corrCovQual = -1;

   if (m.getNPar() > 0) {
      if (cfg.doAsymptotic == 1)
         corrCovQual = calcAsymptoticCorrectedCovariance(pdf, m, data); // Asymptotically correct
      if (cfg.doSumW2 == 1)
         corrCovQual = calcSumW2CorrectedCovariance(pdf, m, nll);
   }

   if (cfg.minos)
      cfg.minosSet ? m.minos(*cfg.minosSet) : m.minos(); // Evaluate errs with Minos

   // Optionally return fit result
   std::unique_ptr<RooFitResult> ret;
   if (cfg.doSave) {
      auto name = std::string("fitresult_") + pdf.GetName() + "_" + data.GetName();
      auto title = std::string("Result of fit of p.d.f. ") + pdf.GetName() + " to dataset " + data.GetName();
      ret = std::unique_ptr<RooFitResult>{m.save(name.c_str(), title.c_str())};
      if ((cfg.doSumW2 == 1 || cfg.doAsymptotic == 1) && m.getNPar() > 0)
         ret->setCovQual(corrCovQual);
   }

   if (cfg.optConst)
      m.optimizeConst(0);
   return ret;
}

std::unique_ptr<RooAbsReal> createNLL(RooAbsPdf &pdf, RooAbsData &data, const RooLinkedList &cmdList)
{
   auto timingScope = std::make_unique<ROOT::Math::Util::TimingScope>(
      [&pdf](std::string const &msg) { oocoutI(&pdf, Fitting) << msg << std::endl; }, "Creation of NLL object took");

   auto baseName = std::string("nll_") + pdf.GetName() + "_" + data.GetName();

   // Select the pdf-specific commands
   RooCmdConfig pc("RooAbsPdf::createNLL(" + std::string(pdf.GetName()) + ")");

   pc.defineString("rangeName", "RangeWithName", 0, "", true);
   pc.defineString("addCoefRange", "SumCoefRange", 0, "");
   pc.defineString("globstag", "GlobalObservablesTag", 0, "");
   pc.defineString("globssource", "GlobalObservablesSource", 0, "data");
   pc.defineDouble("rangeLo", "Range", 0, -999.);
   pc.defineDouble("rangeHi", "Range", 1, -999.);
   pc.defineInt("splitRange", "SplitRange", 0, 0);
   pc.defineInt("ext", "Extended", 0, extendedFitDefault);
   pc.defineInt("numcpu", "NumCPU", 0, 1);
   pc.defineInt("interleave", "NumCPU", 1, 0);
   pc.defineInt("verbose", "Verbose", 0, 0);
   pc.defineInt("optConst", "Optimize", 0, 0);
   pc.defineInt("cloneData", "CloneData", 0, 2);
   pc.defineSet("projDepSet", "ProjectedObservables", 0, nullptr);
   pc.defineSet("cPars", "Constrain", 0, nullptr);
   pc.defineSet("glObs", "GlobalObservables", 0, nullptr);
   pc.defineInt("doOffset", "OffsetLikelihood", 0, 0);
   pc.defineSet("extCons", "ExternalConstraints", 0, nullptr);
   pc.defineInt("EvalBackend", "EvalBackend", 0, static_cast<int>(RooFit::EvalBackend::defaultValue()));
   pc.defineDouble("IntegrateBins", "IntegrateBins", 0, -1.);
   pc.defineMutex("Range", "RangeWithName");
   pc.defineMutex("GlobalObservables", "GlobalObservablesTag");
   pc.defineInt("ModularL", "ModularL", 0, 0);

   // New style likelihoods define parallelization through Parallelize(...) on fitTo or attributes on
   // RooMinimizer::Config.
   pc.defineMutex("ModularL", "NumCPU");

   // New style likelihoods define offsetting on minimizer, not on likelihood
   pc.defineMutex("ModularL", "OffsetLikelihood");

   // Process and check varargs
   pc.process(cmdList);
   if (!pc.ok(true)) {
      return nullptr;
   }

   if (pc.getInt("ModularL")) {
      int lut[3] = {2, 1, 0};
      RooFit::TestStatistics::RooAbsL::Extended ext{
         static_cast<RooFit::TestStatistics::RooAbsL::Extended>(lut[pc.getInt("ext")])};

      RooArgSet cParsSet;
      RooArgSet extConsSet;
      RooArgSet glObsSet;

      if (auto tmp = pc.getSet("cPars"))
         cParsSet.add(*tmp);

      if (auto tmp = pc.getSet("extCons"))
         extConsSet.add(*tmp);

      if (auto tmp = pc.getSet("glObs"))
         glObsSet.add(*tmp);

      const std::string rangeName = pc.getString("globstag", "", false);

      RooFit::TestStatistics::NLLFactory builder{pdf, data};
      builder.Extended(ext)
         .ConstrainedParameters(cParsSet)
         .ExternalConstraints(extConsSet)
         .GlobalObservables(glObsSet)
         .GlobalObservablesTag(rangeName.c_str());

      return std::make_unique<RooFit::TestStatistics::RooRealL>("likelihood", "", builder.build());
   }

   // Decode command line arguments
   const char *rangeName = pc.getString("rangeName", nullptr, true);
   const char *addCoefRangeName = pc.getString("addCoefRange", nullptr, true);
   const bool ext = interpretExtendedCmdArg(pdf, pc.getInt("ext"));

   int splitRange = pc.getInt("splitRange");
   int optConst = pc.getInt("optConst");
   int cloneData = pc.getInt("cloneData");
   auto offset = static_cast<RooFit::OffsetMode>(pc.getInt("doOffset"));

   // If no explicit cloneData command is specified, cloneData is set to true if optimization is activated
   if (cloneData == 2) {
      cloneData = optConst;
   }

   if (pc.hasProcessed("Range")) {
      double rangeLo = pc.getDouble("rangeLo");
      double rangeHi = pc.getDouble("rangeHi");

      // Create range with name 'fit' with above limits on all observables
      RooArgSet obs;
      pdf.getObservables(data.get(), obs);
      for (auto arg : obs) {
         RooRealVar *rrv = dynamic_cast<RooRealVar *>(arg);
         if (rrv)
            rrv->setRange("fit", rangeLo, rangeHi);
      }

      // Set range name to be fitted to "fit"
      rangeName = "fit";
   }

   // Set the fitrange attribute of th PDF, add observables ranges for plotting
   resetFitrangeAttributes(pdf, data, baseName, rangeName, splitRange);

   RooArgSet projDeps;
   auto tmp = pc.getSet("projDepSet");
   if (tmp) {
      projDeps.add(*tmp);
   }

   const std::string globalObservablesSource = pc.getString("globssource", "data", false);
   if (globalObservablesSource != "data" && globalObservablesSource != "model") {
      std::string errMsg = "RooAbsPdf::fitTo: GlobalObservablesSource can only be \"data\" or \"model\"!";
      oocoutE(&pdf, InputArguments) << errMsg << std::endl;
      throw std::invalid_argument(errMsg);
   }
   const bool takeGlobalObservablesFromData = globalObservablesSource == "data";

   // Lambda function to create the correct constraint term for a PDF. In old
   // RooFit, we use this PDF itself as the argument, for the new BatchMode
   // we're passing a clone.
   auto createConstr = [&]() -> std::unique_ptr<RooAbsReal> {
      return createConstraintTerm(baseName + "_constr",                    // name
                                  pdf,                                     // pdf
                                  data,                                    // data
                                  pc.getSet("cPars"),                      // Constrain RooCmdArg
                                  pc.getSet("extCons"),                    // ExternalConstraints RooCmdArg
                                  pc.getSet("glObs"),                      // GlobalObservables RooCmdArg
                                  pc.getString("globstag", nullptr, true), // GlobalObservablesTag RooCmdArg
                                  takeGlobalObservablesFromData);          // From GlobalObservablesSource RooCmdArg
   };

   auto evalBackend = static_cast<RooFit::EvalBackend::Value>(pc.getInt("EvalBackend"));

   // Construct BatchModeNLL if requested
   if (evalBackend != RooFit::EvalBackend::Value::Legacy) {

      // Set the normalization range. We need to do it now, because it will be
      // considered in `compileForNormSet`.
      std::string oldNormRange;
      if (pdf.normRange()) {
         oldNormRange = pdf.normRange();
      }
      pdf.setNormRange(rangeName);

      RooArgSet normSet;
      pdf.getObservables(data.get(), normSet);

      if (dynamic_cast<RooSimultaneous const *>(&pdf)) {
         for (auto i : projDeps) {
            auto res = normSet.find(i->GetName());
            if (res != nullptr) {
               res->setAttribute("__conditional__");
            }
         }
      } else {
         normSet.remove(projDeps);
      }

      pdf.setAttribute("SplitRange", splitRange);
      pdf.setStringAttribute("RangeName", rangeName);

      RooFit::Detail::CompileContext ctx{normSet};
      ctx.setLikelihoodMode(true);
      std::unique_ptr<RooAbsArg> head = pdf.compileForNormSet(normSet, ctx);
      std::unique_ptr<RooAbsPdf> pdfClone = std::unique_ptr<RooAbsPdf>{&dynamic_cast<RooAbsPdf &>(*head.release())};

      // reset attributes
      pdf.setAttribute("SplitRange", false);
      pdf.setStringAttribute("RangeName", nullptr);

      // Reset the normalization range
      pdf.setNormRange(oldNormRange.c_str());

      if (addCoefRangeName) {
         oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                                  << ") fixing interpretation of coefficients of any component to range "
                                  << addCoefRangeName << "\n";
         pdfClone->fixAddCoefRange(addCoefRangeName, false);
      }

      std::unique_ptr<RooAbsReal> compiledConstr;
      if (std::unique_ptr<RooAbsReal> constr = createConstr()) {
         compiledConstr = RooFit::Detail::compileForNormSet(*constr, *data.get());
         compiledConstr->addOwnedComponents(std::move(constr));
      }

      auto nll = createNLLNew(*pdfClone, data, std::move(compiledConstr), rangeName ? rangeName : "", projDeps, ext,
                              pc.getDouble("IntegrateBins"), offset);

      const double correction = pdfClone->getCorrection();

      if (correction > 0) {
         oocoutI(&pdf, Fitting) << "[FitHelpers] Detected correction term from RooAbsPdf::getCorrection(). "
                                << "Adding penalty to NLL." << std::endl;

         // Convert the multiplicative correction to an additive term in -log L
         auto penaltyTerm = std::make_unique<RooConstVar>((baseName + "_Penalty").c_str(),
                                                          "Penalty term from getCorrection()", correction);

         // add penalty and NLL
         auto correctedNLL = std::make_unique<RooAddition>((baseName + "_corrected").c_str(), "NLL + penalty",
                                                           RooArgSet{*nll, *penaltyTerm});

         // transfer ownership of terms
         correctedNLL->addOwnedComponents(std::move(nll), std::move(penaltyTerm));
         nll = std::move(correctedNLL);
      }

      auto nllWrapper = std::make_unique<RooEvaluatorWrapper>(
         *nll, &data, evalBackend == RooFit::EvalBackend::Value::Cuda, rangeName ? rangeName : "", pdfClone.get(),
         takeGlobalObservablesFromData);

      // We destroy the timing scrope for createNLL prematurely, because we
      // separately measure the time for jitting and gradient creation
      // inside the RooFuncWrapper.
      timingScope.reset();

      if (evalBackend == RooFit::EvalBackend::Value::Codegen) {
         nllWrapper->generateGradient();
      }
      if (evalBackend == RooFit::EvalBackend::Value::CodegenNoGrad) {
         nllWrapper->setUseGeneratedFunctionCode(true);
      }

      nllWrapper->addOwnedComponents(std::move(nll));
      nllWrapper->addOwnedComponents(std::move(pdfClone));

      return nllWrapper;
   }

   std::unique_ptr<RooAbsReal> nll;

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   bool verbose = pc.getInt("verbose");

   int numcpu = pc.getInt("numcpu");
   int numcpu_strategy = pc.getInt("interleave");
   // strategy 3 works only for RooSimultaneous.
   if (numcpu_strategy == 3 && !pdf.InheritsFrom("RooSimultaneous")) {
      oocoutW(&pdf, Minimization) << "Cannot use a NumCpu Strategy = 3 when the pdf is not a RooSimultaneous, "
                                     "falling back to default strategy = 0"
                                  << std::endl;
      numcpu_strategy = 0;
   }
   RooFit::MPSplit interl = (RooFit::MPSplit)numcpu_strategy;

   auto binnedLInfo = RooHelpers::getBinnedL(pdf);
   RooAbsPdf &actualPdf = binnedLInfo.binnedPdf ? *binnedLInfo.binnedPdf : pdf;

   // Construct NLL
   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);
   RooAbsTestStatistic::Configuration cfg;
   cfg.addCoefRangeName = addCoefRangeName ? addCoefRangeName : "";
   cfg.nCPU = numcpu;
   cfg.interleave = interl;
   cfg.verbose = verbose;
   cfg.splitCutRange = static_cast<bool>(splitRange);
   cfg.cloneInputData = static_cast<bool>(cloneData);
   cfg.integrateOverBinsPrecision = pc.getDouble("IntegrateBins");
   cfg.binnedL = binnedLInfo.isBinnedL;
   cfg.takeGlobalObservablesFromData = takeGlobalObservablesFromData;
   cfg.rangeName = rangeName ? rangeName : "";
   auto nllVar = std::make_unique<RooNLLVar>(baseName.c_str(), "-log(likelihood)", actualPdf, data, projDeps, ext, cfg);
   nllVar->enableBinOffsetting(offset == RooFit::OffsetMode::Bin);
   nll = std::move(nllVar);
   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors);

   // Include constraints, if any, in likelihood
   if (std::unique_ptr<RooAbsReal> constraintTerm = createConstr()) {

      // Even though it is technically only required when the computation graph
      // is changed because global observables are taken from data, it is safer
      // to clone the constraint model in general to reset the normalization
      // integral caches and avoid ASAN build failures (the PDF of the main
      // measurement is cloned too anyway, so not much overhead). This can be
      // reconsidered after the caching of normalization sets by pointer is changed
      // to a more memory-safe solution.
      constraintTerm = RooHelpers::cloneTreeWithSameParameters(*constraintTerm, data.get());

      // Redirect the global observables to the ones from the dataset if applicable.
      constraintTerm->setData(data, false);

      // The computation graph for the constraints is very small, no need to do
      // the tracking of clean and dirty nodes here.
      constraintTerm->setOperMode(RooAbsArg::ADirty);

      auto orignll = std::move(nll);
      nll = std::make_unique<RooAddition>((baseName + "_with_constr").c_str(), "nllWithCons",
                                          RooArgSet(*orignll, *constraintTerm));
      nll->addOwnedComponents(std::move(orignll), std::move(constraintTerm));
   }

   if (optConst) {
      nll->constOptimizeTestStatistic(RooAbsArg::Activate, optConst > 1);
   }

   if (offset == RooFit::OffsetMode::Initial) {
      nll->enableOffsetting(true);
   }

   if (const double correction = pdf.getCorrection(); correction > 0) {
      oocoutI(&pdf, Fitting) << "[FitHelpers] Detected correction term from RooAbsPdf::getCorrection(). "
                             << "Adding penalty to NLL." << std::endl;

      // Convert the multiplicative correction to an additive term in -log L
      auto penaltyTerm = std::make_unique<RooConstVar>((baseName + "_Penalty").c_str(),
                                                       "Penalty term from getCorrection()", correction);

      auto correctedNLL = std::make_unique<RooAddition>(
         // add penalty and NLL
         (baseName + "_corrected").c_str(), "NLL + penalty", RooArgSet(*nll, *penaltyTerm));

      // transfer ownership of terms
      correctedNLL->addOwnedComponents(std::move(nll), std::move(penaltyTerm));
      nll = std::move(correctedNLL);
   }
#else
   throw std::runtime_error("RooFit was not built with the legacy evaluation backend");
#endif

   return nll;
}

std::unique_ptr<RooAbsReal> createChi2(RooAbsReal &real, RooAbsData &data, const RooLinkedList &cmdList)
{
#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   const bool isDataHist = dynamic_cast<RooDataHist const *>(&data);

   RooCmdConfig pc("createChi2(" + std::string(real.GetName()) + ")");

   pc.defineInt("numcpu", "NumCPU", 0, 1);
   pc.defineInt("verbose", "Verbose", 0, 0);

   RooAbsTestStatistic::Configuration cfg;

   if (isDataHist) {
      // Construct Chi2
      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);
      std::string baseName = "chi2_" + std::string(real.GetName()) + "_" + data.GetName();

      // Clear possible range attributes from previous fits.
      real.removeStringAttribute("fitrange");

      pc.defineInt("etype", "DataError", 0, (Int_t)RooDataHist::Auto);
      pc.defineInt("extended", "Extended", 0, extendedFitDefault);
      pc.defineInt("split_range", "SplitRange", 0, 0);
      pc.defineDouble("integrate_bins", "IntegrateBins", 0, -1);
      pc.defineString("addCoefRange", "SumCoefRange", 0, "");
      pc.allowUndefined();

      pc.process(cmdList);
      if (!pc.ok(true)) {
         return nullptr;
      }

      bool extended = false;
      if (auto pdf = dynamic_cast<RooAbsPdf const *>(&real)) {
         extended = interpretExtendedCmdArg(*pdf, pc.getInt("extended"));
      }

      RooDataHist::ErrorType etype = static_cast<RooDataHist::ErrorType>(pc.getInt("etype"));

      const char *rangeName = pc.getString("rangeName", nullptr, true);
      const char *addCoefRangeName = pc.getString("addCoefRange", nullptr, true);

      cfg.rangeName = rangeName ? rangeName : "";
      cfg.nCPU = pc.getInt("numcpu");
      cfg.interleave = RooFit::Interleave;
      cfg.verbose = static_cast<bool>(pc.getInt("verbose"));
      cfg.cloneInputData = false;
      cfg.integrateOverBinsPrecision = pc.getDouble("integrate_bins");
      cfg.addCoefRangeName = addCoefRangeName ? addCoefRangeName : "";
      cfg.splitCutRange = static_cast<bool>(pc.getInt("split_range"));
      auto chi2 = std::make_unique<RooChi2Var>(baseName.c_str(), baseName.c_str(), real,
                                               static_cast<RooDataHist &>(data), extended, etype, cfg);

      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors);

      return chi2;
   } else {
      pc.defineInt("integrate", "Integrate", 0, 0);
      pc.defineObject("yvar", "YVar", 0, nullptr);
      pc.defineString("rangeName", "RangeWithName", 0, "", true);
      pc.defineInt("interleave", "NumCPU", 1, 0);

      // Process and check varargs
      pc.process(cmdList);
      if (!pc.ok(true)) {
         return nullptr;
      }

      // Decode command line arguments
      bool integrate = pc.getInt("integrate");
      RooRealVar *yvar = static_cast<RooRealVar *>(pc.getObject("yvar"));
      const char *rangeName = pc.getString("rangeName", nullptr, true);
      Int_t numcpu = pc.getInt("numcpu");
      Int_t numcpu_strategy = pc.getInt("interleave");
      // strategy 3 works only for RooSimultaneous.
      if (numcpu_strategy == 3 && !real.InheritsFrom("RooSimultaneous")) {
         oocoutW(&real, Minimization) << "Cannot use a NumCpu Strategy = 3 when the pdf is not a RooSimultaneous, "
                                         "falling back to default strategy = 0"
                                      << std::endl;
         numcpu_strategy = 0;
      }
      RooFit::MPSplit interl = (RooFit::MPSplit)numcpu_strategy;
      bool verbose = pc.getInt("verbose");

      cfg.rangeName = rangeName ? rangeName : "";
      cfg.nCPU = numcpu;
      cfg.interleave = interl;
      cfg.verbose = verbose;
      cfg.verbose = false;

      std::string name = "chi2_" + std::string(real.GetName()) + "_" + data.GetName();

      return std::make_unique<RooXYChi2Var>(name.c_str(), name.c_str(), real, static_cast<RooDataSet &>(data), yvar,
                                            integrate, cfg);
   }
#else
   throw std::runtime_error("createChi2() is not supported without the legacy evaluation backend");
   return nullptr;
#endif
}

std::unique_ptr<RooFitResult> fitTo(RooAbsReal &real, RooAbsData &data, const RooLinkedList &cmdList, bool chi2)
{
   const bool isDataHist = dynamic_cast<RooDataHist const *>(&data);

   RooCmdConfig pc("fitTo(" + std::string(real.GetName()) + ")");

   RooLinkedList fitCmdList(cmdList);
   std::string nllCmdListString;
   if (!chi2) {
      nllCmdListString = "ProjectedObservables,Extended,Range,"
                         "RangeWithName,SumCoefRange,NumCPU,SplitRange,Constrained,Constrain,ExternalConstraints,"
                         "CloneData,GlobalObservables,GlobalObservablesSource,GlobalObservablesTag,"
                         "EvalBackend,IntegrateBins,ModularL";

      if (!cmdList.FindObject("ModularL") || static_cast<RooCmdArg *>(cmdList.FindObject("ModularL"))->getInt(0) == 0) {
         nllCmdListString += ",OffsetLikelihood";
      }
   } else {
      auto createChi2DataHistCmdArgs = "Range,RangeWithName,NumCPU,Optimize,IntegrateBins,ProjectedObservables,"
                                       "AddCoefRange,SplitRange,DataError,Extended";
      auto createChi2DataSetCmdArgs = "YVar,Integrate,RangeWithName,NumCPU,Verbose";
      nllCmdListString += isDataHist ? createChi2DataHistCmdArgs : createChi2DataSetCmdArgs;
   }

   RooLinkedList nllCmdList = pc.filterCmdList(fitCmdList, nllCmdListString.c_str());

   pc.defineDouble("prefit", "Prefit", 0, 0);
   defineMinimizationOptions(pc);

   // Process and check varargs
   pc.process(fitCmdList);
   if (!pc.ok(true)) {
      return nullptr;
   }

   // TimingAnalysis works only for RooSimultaneous.
   if (pc.getInt("timingAnalysis") && !real.InheritsFrom("RooSimultaneous")) {
      oocoutW(&real, Minimization) << "The timingAnalysis feature was built for minimization with RooSimultaneous "
                                      "and is not implemented for other PDF's. Please create a RooSimultaneous to "
                                      "enable this feature."
                                   << std::endl;
   }

   // Decode command line arguments
   double prefit = pc.getDouble("prefit");

   if (prefit != 0) {
      size_t nEvents = static_cast<size_t>(prefit * data.numEntries());
      if (prefit > 0.5 || nEvents < 100) {
         oocoutW(&real, InputArguments) << "PrefitDataFraction should be in suitable range."
                                        << "With the current PrefitDataFraction=" << prefit
                                        << ", the number of events would be " << nEvents << " out of "
                                        << data.numEntries() << ". Skipping prefit..." << std::endl;
      } else {
         size_t step = data.numEntries() / nEvents;

         RooDataSet tiny("tiny", "tiny", *data.get(), data.isWeighted() ? RooFit::WeightVar() : RooCmdArg());

         for (int i = 0; i < data.numEntries(); i += step) {
            const RooArgSet *event = data.get(i);
            tiny.add(*event, data.weight());
         }
         RooLinkedList tinyCmdList(cmdList);
         pc.filterCmdList(tinyCmdList, "Prefit,Hesse,Minos,Verbose,Save,Timer");
         RooCmdArg hesse_option = RooFit::Hesse(false);
         RooCmdArg print_option = RooFit::PrintLevel(-1);

         tinyCmdList.Add(&hesse_option);
         tinyCmdList.Add(&print_option);

         fitTo(real, tiny, tinyCmdList, chi2);
      }
   }

   RooCmdArg modularL_option;
   if (pc.getInt("parallelize") != 0 || pc.getInt("enableParallelGradient") || pc.getInt("enableParallelDescent")) {
      // Set to new style likelihood if parallelization is requested
      modularL_option = RooFit::ModularL(true);
      nllCmdList.Add(&modularL_option);
   }

   std::unique_ptr<RooAbsReal> nll;
   if (chi2) {
      nll = std::unique_ptr<RooAbsReal>{isDataHist ? real.createChi2(static_cast<RooDataHist &>(data), nllCmdList)
                                                   : real.createChi2(static_cast<RooDataSet &>(data), nllCmdList)};
   } else {
      nll = std::unique_ptr<RooAbsReal>{dynamic_cast<RooAbsPdf &>(real).createNLL(data, nllCmdList)};
   }

   return RooFit::FitHelpers::minimize(real, *nll, data, pc);
}

} // namespace FitHelpers
} // namespace RooFit

/// \endcond
