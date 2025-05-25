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

#include <array>


/**
 * \class RooNormalizedPdf
 *
 * A RooNormalizedPdf wraps a pdf divided by its integral for a given
 * normalization set into a new self-normalized pdf.
 */

namespace RooFit {
namespace Detail {

void RooNormalizedPdf::doEval(RooFit::EvalContext &ctx) const
{
   auto nums = ctx.at(_pdf);
   auto integralSpan = ctx.at(_normIntegral);

   // We use the extraArgs as output parameter to count evaluation errors.
   std::array<double, 3> extraArgs{0.0, 0.0, 0.0};

   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::NormalizedPdf, ctx.output(), {nums, integralSpan},
                            extraArgs);

   std::size_t nEvalErrorsType0 = extraArgs[0];
   std::size_t nEvalErrorsType1 = extraArgs[1];
   std::size_t nEvalErrorsType2 = extraArgs[2];

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

} // namespace Detail
} // namespace RooFit
