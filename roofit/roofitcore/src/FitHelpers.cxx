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
#include <RooCmdConfig.h>
#include <RooDerivative.h>
#include <RooFitResult.h>
#include <RooLinkedList.h>
#include <RooMinimizer.h>
#include <RooRealVar.h>

#include <Math/CholeskyDecomp.h>

namespace RooFit {
namespace FitHelpers {

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
   // Calculated corrected errors for weighted likelihood fits
   std::unique_ptr<RooFitResult> rw(minimizer.save());
   // Weighted inverse Hessian matrix
   const TMatrixDSym &matV = rw->covarianceMatrix();
   oocoutI(&pdf, Fitting)
      << "RooAbsPdf::fitTo(" << pdf.GetName()
      << ") Calculating covariance matrix according to the asymptotically correct approach. If you find this "
         "method useful please consider citing https://arxiv.org/abs/1911.01303.\n";

   // Initialise matrix containing first derivatives
   auto nFloatPars = rw->floatParsFinal().getSize();
   TMatrixDSym num(nFloatPars);
   for (int k = 0; k < nFloatPars; k++) {
      for (int l = 0; l < nFloatPars; l++) {
         num(k, l) = 0.0;
      }
   }
   RooArgSet obs;
   pdf.getObservables(data.get(), obs);
   // Create derivative objects
   std::vector<std::unique_ptr<RooDerivative>> derivatives;
   const RooArgList &floated = rw->floatParsFinal();
   std::unique_ptr<RooArgSet> floatingparams{
      static_cast<RooArgSet *>(pdf.getParameters(data)->selectByAttrib("Constant", false))};
   for (const auto paramresult : floated) {
      auto paraminternal = static_cast<RooRealVar *>(floatingparams->find(*paramresult));
      assert(floatingparams->find(*paramresult)->IsA() == RooRealVar::Class());
      derivatives.emplace_back(pdf.derivative(*paraminternal, obs, 1));
   }

   // Loop over data
   for (int j = 0; j < data.numEntries(); j++) {
      // Sets obs to current data point, this is where the pdf will be evaluated
      obs.assign(*data.get(j));
      // Determine first derivatives
      std::vector<double> diffs(floated.getSize(), 0.0);
      for (int k = 0; k < floated.getSize(); k++) {
         const auto paramresult = static_cast<RooRealVar *>(floated.at(k));
         auto paraminternal = static_cast<RooRealVar *>(floatingparams->find(*paramresult));
         // first derivative to parameter k at best estimate point for this measurement
         double diff = derivatives[k]->getVal();
         // need to reset to best fit point after differentiation
         *paraminternal = paramresult->getVal();
         diffs[k] = diff;
      }
      // Fill numerator matrix
      double prob = pdf.getVal(&obs);
      for (int k = 0; k < floated.getSize(); k++) {
         for (int l = 0; l < floated.getSize(); l++) {
            num(k, l) += data.weightSquared() * diffs[k] * diffs[l] / (prob * prob);
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

namespace {

/// Configuration struct for RooAbsPdf::minimizeNLL with all the default values
/// that also should be taked as the default values for RooAbsPdf::fitTo.
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

} // namespace

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
         corrCovQual = RooFit::FitHelpers::calcAsymptoticCorrectedCovariance(pdf, m, data); // Asymptotically correct
      if (cfg.doSumW2 == 1)
         corrCovQual = RooFit::FitHelpers::calcSumW2CorrectedCovariance(pdf, m, nll);
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

} // namespace FitHelpers
} // namespace RooFit
