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

#ifndef RooFit_FitHelpers_h
#define RooFit_FitHelpers_h

#include <memory>

class RooAbsData;
class RooAbsPdf;
class RooAbsReal;
class RooCmdConfig;
class RooFitResult;
class RooLinkedList;

namespace RooFit {
namespace FitHelpers {

void defineMinimizationOptions(RooCmdConfig &pc);

std::unique_ptr<RooFitResult> minimize(RooAbsReal &model, RooAbsReal &nll, RooAbsData const &data, RooCmdConfig const &pc);

std::unique_ptr<RooAbsReal> createNLL(RooAbsPdf &pdf, RooAbsData &data, const RooLinkedList &cmdList);
std::unique_ptr<RooAbsReal> createChi2(RooAbsReal &real, RooAbsData &data, const RooLinkedList &cmdList);

std::unique_ptr<RooFitResult> fitTo(RooAbsReal &pdf, RooAbsData &data, const RooLinkedList &cmdList, bool chi2);

} // namespace FitHelpers
} // namespace RooFit

#endif

/// \endcond
