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
class RooCmdConfig;
class RooFitResult;
class RooLinkedList;
class RooMinimizer;

namespace RooFit {
namespace FitHelpers {

int calcAsymptoticCorrectedCovariance(RooAbsPdf &pdf, RooMinimizer &minimizer, RooAbsData const &data);
int calcSumW2CorrectedCovariance(RooAbsPdf const &pdf, RooMinimizer &minimizer, RooAbsReal &nll);

void defineMinimizationOptions(RooCmdConfig &pc);
std::unique_ptr<RooFitResult> minimize(RooAbsReal &model, RooAbsReal &nll, RooAbsData const &data, RooCmdConfig const &pc);

constexpr int extendedFitDefault = 2;

} // namespace FitHelpers
} // namespace RooFit

#endif
