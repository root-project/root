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

#include <RooBatchCompute.h>
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

void DataMap::setConfig(RooAbsArg const *arg, RooBatchCompute::Config const &config)
{
   if (!arg->hasDataToken())
      return;
   std::size_t idx = arg->dataToken();
   _cfgs[idx] = config;
}

RooBatchCompute::Config DataMap::config(RooAbsArg const *arg) const
{
   if (!arg->hasDataToken()) {
      return {};
   }
   std::size_t idx = arg->dataToken();
   return _cfgs[idx];
}

void DataMap::resize(std::size_t n)
{
   _cfgs.resize(n);
   _dataMap.resize(n);
}

} // namespace Detail
} // namespace RooFit
