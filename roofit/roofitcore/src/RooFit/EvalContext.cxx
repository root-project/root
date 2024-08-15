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

#include <RooFit/EvalContext.h>

#include <RooBatchCompute.h>
#include <RooRealVar.h>

#include <algorithm>
#include <stdexcept>

namespace {

// To avoid deleted move assignment.
template <class T>
void assignSpan(std::span<T> &to, std::span<T> const &from)
{
   to = from;
}

} // namespace

namespace RooFit {

std::span<const double> EvalContext::at(RooAbsArg const *arg, RooAbsArg const * /*caller*/)
{
   std::span<const double> out;

   if (!arg->hasDataToken()) {
      auto var = static_cast<RooRealVar const *>(arg);
      assignSpan(out, {&var->_value, 1});
   } else {
      std::size_t idx = arg->dataToken();
      out = _ctx[idx];
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

void EvalContext::setConfig(RooAbsArg const *arg, RooBatchCompute::Config const &config)
{
   if (!arg->hasDataToken())
      return;
   std::size_t idx = arg->dataToken();
   _cfgs[idx] = config;
}

RooBatchCompute::Config EvalContext::config(RooAbsArg const *arg) const
{
   if (!arg->hasDataToken()) {
      return {};
   }
   std::size_t idx = arg->dataToken();
   return _cfgs[idx];
}

void EvalContext::resize(std::size_t n)
{
   _cfgs.resize(n);
   _ctx.resize(n);
}

/// \brief Sets the output value with an offset.
///
/// This function sets the output value with an offset for the given argument.
/// It should only be used in reducer nodes. Depending on the current
/// OffsetMode, the result will either be just the value, the value minus the
/// offset, of just the offset.
///
/// \param arg Pointer to the RooAbsArg object.
/// \param val The value to be set.
/// \param offset The offset value.
///
/// \throws std::runtime_error if the argument is not a reducer node.
void EvalContext::setOutputWithOffset(RooAbsArg const *arg, ROOT::Math::KahanSum<double> val,
                                      ROOT::Math::KahanSum<double> const &offset)
{
   if (!arg->isReducerNode()) {
      throw std::runtime_error("You can only use setOutputWithOffset() in reducer nodes!");
   }
   if (_offsetMode == OffsetMode::WithOffset) {
      val -= offset;
   } else if (_offsetMode == OffsetMode::OnlyOffset) {
      val = offset;
   }
   const_cast<double *>(_ctx[arg->dataToken()].data())[0] = val.Sum();
}

} // namespace RooFit
