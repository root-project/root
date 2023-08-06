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

#include "CudaHelpers.h"

#include <RooFit/Detail/CudaInterface.h>

/// Measure the time for transfering data from host to device and vice-versa.
std::pair<std::chrono::microseconds, std::chrono::microseconds> RooFit::CUDAHelpers::memcpyBenchmark(std::size_t nBytes)
{
   using namespace std::chrono;
   namespace CudaInterface = RooFit::Detail::CudaInterface;

   std::pair<std::chrono::microseconds, std::chrono::microseconds> ret;
   char *hostArr = CudaInterface::cudaMallocHost<char>(nBytes);
   char *deviArr = CudaInterface::cudaMalloc<char>(nBytes);
   constexpr std::size_t nIterations = 5;
   for (std::size_t i = 0; i < nIterations; i++) {
      auto start = steady_clock::now();
      CudaInterface::memcpyToCUDA(deviArr, hostArr, nBytes);
      ret.first += duration_cast<microseconds>(steady_clock::now() - start);
      start = steady_clock::now();
      CudaInterface::memcpyToCPU(hostArr, deviArr, nBytes);
      ret.second += duration_cast<microseconds>(steady_clock::now() - start);
   }
   CudaInterface::cudaFreeHost(hostArr);
   CudaInterface::cudaFree(deviArr);
   ret.first /= nIterations;
   ret.second /= nIterations;
   return ret;
}
