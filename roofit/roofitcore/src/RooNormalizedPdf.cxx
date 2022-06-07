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

#include "RooNormalizedPdf.h"

/**
 * \class RooNormalizedPdf
 *
 * A RooNormalizedPdf wraps a pdf divided by its integral for a given
 * normalization set into a new self-normalized pdf.
 */

void RooNormalizedPdf::computeBatch(cudaStream_t * /*stream*/, double *output, size_t nEvents,
                                    RooFit::Detail::DataMap const& dataMap) const
{
   auto nums = dataMap.at(_pdf);
   auto integralSpan = dataMap.at(_normIntegral);

   if (integralSpan.size() == 1) {
      for (std::size_t i = 0; i < nEvents; ++i) {
         output[i] = normalizeWithNaNPacking(nums[i], integralSpan[0]);
      }
   } else {
      assert(integralSpan.size() == nEvents);
      for (std::size_t i = 0; i < nEvents; ++i) {
         output[i] = normalizeWithNaNPacking(nums[i], integralSpan[i]);
      }
   }
}
