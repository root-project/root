/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_BatchModeDataHelpers_h
#define RooFit_BatchModeDataHelpers_h

#include <RooFit/EvalContext.h>

#include <ROOT/RSpan.hxx>

#include <functional>
#include <map>
#include <stack>
#include <vector>

class RooAbsData;
class RooSimultaneous;

namespace RooFit::BatchModeDataHelpers {

std::map<RooFit::Detail::DataKey, std::span<const double>>
getDataSpans(RooAbsData const &data, std::string const &rangeName, RooSimultaneous const *simPdf, bool skipZeroWeights,
             bool takeGlobalObservablesFromData, std::stack<std::vector<double>> &buffers);

std::map<RooFit::Detail::DataKey, std::size_t>
determineOutputSizes(RooAbsArg const &topNode, std::function<int(RooFit::Detail::DataKey)> const &inputSizeFunc);

} // namespace RooFit::BatchModeDataHelpers

#endif

/// \endcond
