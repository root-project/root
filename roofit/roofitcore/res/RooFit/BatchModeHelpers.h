/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_BatchModeHelpers_h
#define RooFit_BatchModeHelpers_h

#include <memory>
#include <string>

class RooAbsData;
class RooAbsPdf;
class RooAbsReal;
class RooArgSet;

namespace RooFit {
namespace BatchModeHelpers {

RooAbsReal *createNLL(RooAbsPdf &pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
                      std::string const &rangeName, std::string const &addCoefRangeName, RooArgSet const &projDeps,
                      bool isExtended, double integrateOverBinsPrecision, int batchMode);

}
} // namespace RooFit

#endif
