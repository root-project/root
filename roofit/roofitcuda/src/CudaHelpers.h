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

#ifndef RooFit_CUDAHelpers_h
#define RooFit_CUDAHelpers_h

#include <chrono>
#include <utility>

namespace RooFit {
namespace CUDAHelpers {

std::pair<std::chrono::microseconds, std::chrono::microseconds> memcpyBenchmark(std::size_t nBytes);

} // namespace CUDAHelpers
} // namespace RooFit

#endif
