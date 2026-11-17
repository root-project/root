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

#include "RooFit/Detail/RooNormalizedPdf.h"

#include "RooBatchCompute.h"
#include "RooFitImplHelpers.h"
#include "RooNaNPacker.h"

#include <array>
#include <cmath>
#include <limits>

/**
 * \class RooNormalizedPdf
 *
 * A RooNormalizedPdf wraps a pdf divided by its integral for a given
 * normalization set into a new self-normalized pdf.
 */

namespace {

/// Logarithm of the normalized pdf, with the same evaluation errors as the
/// NormalizedPdf compute function. The input modes are template parameters,
/// so that no logarithms are evaluated speculatively.
template <bool NumIsLog, bool ScalarDen>
void evalLogNormalized(std::span<double> output, std::span<const double> nums, std::span<const double> dens,
                       std::array<double, 3> &errorCounts)
{
   constexpr double negInf = -std::numeric_limits<double>::infinity();
   const double scalarLogDen = ScalarDen ? std::log(dens[0]) : 0.;
   for (std::size_t i = 0; i < output.size(); ++i) {
      const double num = nums.size() > 1 ? nums[i] : nums[0];
      const double den = ScalarDen ? dens[0] : dens[i];
      const bool numIsZero = NumIsLog ? num == negInf : num == 0.;
      const bool numIsNegative = !NumIsLog && num < 0.;
      if (den < 0. || (den == 0. && !numIsZero)) {
         output[i] = RooNaNPacker::packFloatIntoNaN(-den + (numIsNegative ? -num : 0.));
         ++errorCounts[0];
      } else if (numIsNegative) {
         output[i] = RooNaNPacker::packFloatIntoNaN(-num);
         ++errorCounts[1];
      } else if (std::isnan(num)) {
         output[i] = num;
         ++errorCounts[2];
      } else if (numIsZero) {
         output[i] = negInf;
      } else {
         const double logNum = NumIsLog ? num : std::log(num);
         const double logDen = ScalarDen ? scalarLogDen : std::log(den);
         output[i] = logNum - logDen;
      }
   }
}

} // namespace

namespace RooFit::Detail {

void RooNormalizedPdf::doEval(RooFit::EvalContext &ctx) const
{
   auto nums = ctx.at(_pdf);
   auto integralSpan = ctx.at(_normIntegral);

   // We use the extraArgs as output parameter to count evaluation errors.
   _evalErrorCounts = {};

   if (ctx.outputsLogarithm(this)) {
      // The pdf might provide its logarithm already.
      const bool numIsLog = ctx.outputsLogarithm(&*_pdf);
      const bool scalarDen = integralSpan.size() == 1;
      auto evalLog = numIsLog ? (scalarDen ? evalLogNormalized<true, true> : evalLogNormalized<true, false>)
                              : (scalarDen ? evalLogNormalized<false, true> : evalLogNormalized<false, false>);
      evalLog(ctx.output(), nums, integralSpan, _evalErrorCounts);
      // The logarithm is only requested in CPU evaluation.
      logEvalErrorCounts();
      return;
   }

   auto config = ctx.config(this);
   RooBatchCompute::compute(config, RooBatchCompute::NormalizedPdf, ctx.output(), {nums, integralSpan},
                            _evalErrorCounts);

   if (config.useCuda()) {
      // In CUDA mode, the counters are read back from the GPU without
      // synchronizing the stream: they only arrive in _evalErrorCounts with
      // the synchronization at the end of the evaluation of the full
      // computation graph, so the logging has to be deferred until then.
      ctx.deferAction([this] { logEvalErrorCounts(); });
   } else {
      logEvalErrorCounts();
   }
}

void RooNormalizedPdf::logEvalErrorCounts() const
{
   const std::size_t nEvalErrorsType0 = _evalErrorCounts[0];
   const std::size_t nEvalErrorsType1 = _evalErrorCounts[1];
   const std::size_t nEvalErrorsType2 = _evalErrorCounts[2];

   for (std::size_t i = 0; i < nEvalErrorsType0; ++i) {
      logEvalError("p.d.f normalization integral is zero or negative");
   }
   for (std::size_t i = 0; i < nEvalErrorsType1; ++i) {
      logEvalError("p.d.f value is less than zero, trying to recover");
   }
   for (std::size_t i = 0; i < nEvalErrorsType2; ++i) {
      logEvalError("p.d.f value is Not-a-Number");
   }
}

double RooNormalizedPdf::getValV(const RooArgSet * /*normSet*/) const
{
   return normalizeWithNaNPacking(*_pdf, _pdf->getVal(), _normIntegral->getVal());
}

} // namespace RooFit::Detail
