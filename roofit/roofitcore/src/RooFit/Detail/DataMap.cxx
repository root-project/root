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

#include <algorithm>

namespace {

// To avoid deleted move assignment.
template <class T>
void assignSpan(std::span<T> &to, std::span<T> const &from)
{
   to = from;
}

} // namespace

namespace RooFit {
namespace Detail {

std::span<const double> DataMap::at(RooAbsArg const *arg, RooAbsArg const * /*caller*/)
{
   std::span<const double> out;

   if (!arg->hasDataToken()) {
      auto var = static_cast<RooRealVar const *>(arg);
      assignSpan(out, {&var->_value, 1});
   } else {
      std::size_t idx = arg->dataToken();
      out = _dataMap[idx];
   }

   if (!_enableVectorBuffers || out.size() != 1) {
      return out;
   }

   if (_bufferIdx == _buffers.size()) {
      _buffers.emplace_back(RooBatchCompute::bufferSize);
   }

   double *buffer = _buffers[_bufferIdx].data();

   std::fill_n(buffer, RooBatchCompute::bufferSize, out[0]);
   assignSpan(out, {buffer, 1});

   ++_bufferIdx;

   return out;
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
