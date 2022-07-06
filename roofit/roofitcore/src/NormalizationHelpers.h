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
#include <RooFit/Detail/DataMap.h>

#include <memory>
#include <unordered_map>
#include <vector>

class RooAbsArg;

namespace RooFit {

class NormalizationIntegralUnfolder {
public:
   NormalizationIntegralUnfolder(RooAbsArg const &topNode, RooArgSet const &normSet);
   ~NormalizationIntegralUnfolder();

   inline RooAbsArg &arg() const { return *_arg; }

private:
   std::unique_ptr<RooAbsArg> _topNodeWrapper;
   RooAbsArg *_arg = nullptr;
   std::unordered_map<RooFit::Detail::DataKey, RooArgSet *> _normSets;
   RooArgSet _replacedArgs;
   bool _normSetWasEmpty;
};

} // namespace RooFit

#endif
