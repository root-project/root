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

#include <RooGlobalFunc.h>

#include <memory>
#include <string>

class RooAbsData;
class RooAbsPdf;
class RooAbsReal;
class RooArgSet;

namespace RooFit {
namespace BatchModeHelpers {

std::unique_ptr<RooAbsReal> createNLL(RooAbsPdf &pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
                                      std::string const &rangeName, std::string const &addCoefRangeName,
                                      RooArgSet const &projDeps, bool isExtended, double integrateOverBinsPrecision,
                                      RooFit::BatchModeOption batchMode);

// Little wrapper to use a TNamed directly as a RooBatchCompute DataKey.
class NamePtrWrapper {
public:
   NamePtrWrapper(TNamed const *namePtr) : _namePtr(namePtr) {}
   operator RooBatchCompute::DataKey() const { return RooBatchCompute::DataKey::create(_namePtr); }

private:
   TNamed const *_namePtr;
};

}
} // namespace RooFit

#endif
