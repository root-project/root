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

#ifndef RooFit_NormalizationHelpers_h
#define RooFit_NormalizationHelpers_h

#include <RooArgSet.h>

#include <memory>
#include <unordered_map>
#include <vector>

class RooAbsArg;

namespace RooFit {

class NormalizationIntegralUnfolder {
public:
   NormalizationIntegralUnfolder(RooAbsArg const &topNode, RooArgSet const &normSet);
   ~NormalizationIntegralUnfolder();

   RooAbsArg const &arg() const;

private:
   std::unique_ptr<RooAbsArg> _topNodeWrapper;
   std::unordered_map<RooAbsArg const *, RooArgSet*> _normSets;
};

} // namespace RooFit

#endif
