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

#ifndef RooFit_FitHelpers_h
#define RooFit_FitHelpers_h

#include <memory>
#include <string>

class RooAbsData;
class RooAbsPdf;
class RooAbsReal;
class RooArgSet;
class RooFitResult;
class RooLinkedList;
class RooMinimizer;

namespace RooFit {
namespace FitHelpers {

/// Configuration struct for RooAbsPdf::minimizeNLL with all the default values
/// that also should be taked as the default values for RooAbsPdf::fitTo.
struct MinimizerConfig {
   double recoverFromNaN = 10.;
   int optConst = 2;
   int verbose = 0;
   int doSave = 0;
   int doTimer = 0;
   int printLevel = 1;
   int strat = 1;
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

int calcAsymptoticCorrectedCovariance(RooAbsPdf &pdf, RooMinimizer &minimizer, RooAbsData const &data);
int calcSumW2CorrectedCovariance(RooAbsPdf const &pdf, RooMinimizer &minimizer, RooAbsReal &nll);
std::unique_ptr<RooFitResult>
minimizeNLL(RooAbsPdf &pdf, RooAbsReal &nll, RooAbsData const &data, MinimizerConfig const &cfg);
std::unique_ptr<RooFitResult> chi2FitDriver(RooAbsReal &fcn, RooLinkedList &cmdList);

constexpr int extendedFitDefault = 2;

} // namespace FitHelpers
} // namespace RooFit

#endif
