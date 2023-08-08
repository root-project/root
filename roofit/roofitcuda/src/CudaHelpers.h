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

#ifndef RooFit_CudaHelpers_h
#define RooFit_CudaHelpers_h

#include <chrono>
#include <utility>

namespace RooFit {
namespace Detail {
/*
 * Contains helper functions that might be useful in the context of using a CUDA GPU.
 *
 * \ingroup RooFitCuda
 */
namespace CudaHelpers {

std::pair<std::chrono::microseconds, std::chrono::microseconds> memcpyBenchmark(std::size_t nBytes);

} // namespace CudaHelpers
} // namespace Detail
} // namespace RooFit

#endif
