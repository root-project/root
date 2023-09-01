/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "CudaHelpers.h"

#include <RooFit/Detail/CudaInterface.h>

/**
 * @brief Measure the time taken to perform memory copy operations between host and device memory.
 * @param nBytes The number of bytes to be copied between host and device memory.
 * @return A pair of durations representing the average time taken for host-to-device and device-to-host copies.
 *
 * This function measures the time taken to copy data between host and device memory using the CUDA API.
 * It performs a series of copy operations and calculates the average time for both directions.
 * The input parameter `nBytes` specifies the size of the data to be copied in bytes.
 * The function returns a pair of durations, where the first duration represents the average time
 * taken for host-to-device copies and the second duration represents the average time taken for
 * device-to-host copies.
 *
 * Example usage:
 * @code
 * auto copyTimes = RooFit::Detail::CudaHelpers::memcpyBenchmark(1024 * 1024); // Measure copy time for 1 MB of data
 * std::cout << "Average Host-to-Device Copy Time: " << copyTimes.first.count() << " microseconds" << std::endl;
 * std::cout << "Average Device-to-Host Copy Time: " << copyTimes.second.count() << " microseconds" << std::endl;
 * @endcode
 */
std::pair<std::chrono::microseconds, std::chrono::microseconds>
RooFit::Detail::CudaHelpers::memcpyBenchmark(std::size_t nBytes)
{
   using namespace std::chrono;
   namespace CudaInterface = RooFit::Detail::CudaInterface;

   std::pair<std::chrono::microseconds, std::chrono::microseconds> ret;
   CudaInterface::PinnedHostArray<char> hostArr{nBytes};
   CudaInterface::DeviceArray<char> deviArr{nBytes};
   constexpr std::size_t nIterations = 5;
   for (std::size_t i = 0; i < nIterations; i++) {
      auto start = steady_clock::now();
      CudaInterface::copyHostToDevice(hostArr.data(), deviArr.data(), nBytes);
      ret.first += duration_cast<microseconds>(steady_clock::now() - start);
      start = steady_clock::now();
      CudaInterface::copyDeviceToHost(deviArr.data(), hostArr.data(), nBytes);
      ret.second += duration_cast<microseconds>(steady_clock::now() - start);
   }
   ret.first /= nIterations;
   ret.second /= nIterations;
   return ret;
}
