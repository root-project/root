/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2023
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/Detail/DataMap.h>

#include <RooRealVar.h>

namespace RooFit {
namespace Detail {

RooSpan<const double> DataMap::at(RooAbsArg const *arg, RooAbsArg const * /*caller*/)
{
   if (!arg->hasDataToken()) {
      auto var = static_cast<RooRealVar const *>(arg);
      return {&var->_value, 1};
   }
   std::size_t idx = arg->dataToken();
   return _dataMap[idx];
}

} // namespace Detail
} // namespace RooFit
