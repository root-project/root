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

#include <array>

/**
 * \class RooNormalizedPdf
 *
 * A RooNormalizedPdf wraps a pdf divided by its integral for a given
 * normalization set into a new self-normalized pdf.
 */

namespace RooFit::Detail {

void RooNormalizedPdf::doEval(RooFit::EvalContext &ctx) const
{
   auto nums = ctx.at(_pdf);
   auto integralSpan = ctx.at(_normIntegral);
   auto config = ctx.config(this);

   // The first three extra args are output parameters to count evaluation
   // errors. The last one tells the LogNormalizedPdf computer whether the pdf
   // values are logarithms already.
   _extraArgs = {0., 0., 0., static_cast<double>(ctx.outputsLogarithm(&*_pdf))};
   const auto computer =
      ctx.outputsLogarithm(this) ? RooBatchCompute::LogNormalizedPdf : RooBatchCompute::NormalizedPdf;
   RooBatchCompute::compute(config, computer, ctx.output(), {nums, integralSpan}, _extraArgs);

   if (config.useCuda()) {
      // In CUDA mode, the counters are read back from the GPU without
      // synchronizing the stream: they only arrive in _extraArgs with the
      // synchronization at the end of the evaluation of the full computation
      // graph, so the logging has to be deferred until then.
      ctx.deferAction([this] { logEvalErrorCounts(); });
   } else {
      logEvalErrorCounts();
   }
}

void RooNormalizedPdf::logEvalErrorCounts() const
{
   const std::size_t nEvalErrorsType0 = _extraArgs[0];
   const std::size_t nEvalErrorsType1 = _extraArgs[1];
   const std::size_t nEvalErrorsType2 = _extraArgs[2];

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
