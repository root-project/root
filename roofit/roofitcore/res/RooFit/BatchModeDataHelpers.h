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

#include <RooSpan.h>

#include <ROOT/RStringView.hxx>

#include <map>
#include <memory>
#include <stack>
#include <vector>

class RooAbsCategory;
class RooAbsData;
class TNamed;

namespace RooFit {
namespace BatchModeDataHelpers {

std::map<const TNamed *, RooSpan<const double>> getDataSpans(RooAbsData const &data, std::string_view rangeName,
                                                             RooAbsCategory const *indexCat,
                                                             std::stack<std::vector<double>> &buffers);

} // namespace BatchModeDataHelpers
} // namespace RooFit

#endif
