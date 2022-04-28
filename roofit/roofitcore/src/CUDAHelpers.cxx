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

#include <RooFit/CUDAHelpers.h>

#include <RooBatchCompute.h>

/// Measure the time for transfering data from host to device and vice-versa.
std::pair<std::chrono::microseconds, std::chrono::microseconds> RooFit::CUDAHelpers::memcpyBenchmark(std::size_t nBytes)
{
   using namespace std::chrono;

   std::pair<std::chrono::microseconds, std::chrono::microseconds> ret;
   auto hostArr = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMallocHost(nBytes));
   auto deviArr = static_cast<double *>(RooBatchCompute::dispatchCUDA->cudaMalloc(nBytes));
   for (int i = 0; i < 5; i++) {
      auto start = steady_clock::now();
      RooBatchCompute::dispatchCUDA->memcpyToCUDA(deviArr, hostArr, nBytes);
      ret.first += duration_cast<microseconds>(steady_clock::now() - start);
      start = steady_clock::now();
      RooBatchCompute::dispatchCUDA->memcpyToCPU(hostArr, deviArr, nBytes);
      ret.second += duration_cast<microseconds>(steady_clock::now() - start);
   }
   RooBatchCompute::dispatchCUDA->cudaFreeHost(hostArr);
   RooBatchCompute::dispatchCUDA->cudaFree(deviArr);
   ret.first /= 5;
   ret.second /= 5;
   return ret;
}
